"""
Multi-agent WhatsApp runtime
- LangGraph create_react_agent with static prompt and static tools
- No Redis, no DB fetch for agent config
- FastAPI server exposing POST /run-agent
- Agent is selected by port so the same infra can serve different assistants
"""

import os
import json
import uuid
import traceback
import urllib.parse
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Annotated
from operator import ior

import requests
import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState

from langchain_core.tools import StructuredTool
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
)
try:
    from langchain_core.messages import messages_from_dict, messages_to_dict
except ImportError:  # pragma: no cover - older langchain_core fallback
    messages_from_dict = None
    messages_to_dict = None
from langchain_openai import ChatOpenAI

from pydantic import create_model, Field, BaseModel, ConfigDict
from langgraph.errors import GraphRecursionError


# ============================================================
# CONFIGURATION - Set these via environment variables
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
PORT = int(os.getenv("PORT", "8001"))
OPENAI_BASE_URL = "https://api.openai.com/v1"


# ============================================================
# AGENT STATE
# ============================================================
class AgentState(MessagesState):
    variables: Annotated[Dict[str, str], ior]
    is_last_step: bool = False
    remaining_steps: int = 0


# ============================================================
# SYSTEM PROMPTS
# ============================================================
PARTSWALE_SYSTEM_PROMPT = r"""You are a WhatsApp assistant for mechanics to request and manage spare parts.

You are a TASK EXECUTION SYSTEM, not a conversational AI.

---

INPUT CONTEXT:

You receive:
- Last 5 WhatsApp messages (incoming + outgoing)
- Current user message
- Available data (requests, quotes, orders)

---


You must:
- Understand user intent
- Extract part requests (single or multiple)
- Ask ONLY required missing details
- Confirm before posting request
- Show quotes when available
- Allow ordering
- Show order updates
- Show the correct menu or fixed state message whenever applicable

---

CRITICAL RULES:

0. LANGUAGE RULE
- Always reply in Hinglish, no matter what language the user speaks
- Convert every user-facing message to natural Hinglish
- Keep the meaning, structure, and state logic of the selected template the same
- Do not output the English template verbatim in live replies unless a brand name, product name, URL, or fixed button label requires it
- Translate only the actual generated reply, not this prompt

1. NO GUESSING
- Never assume missing details
- Only extract what user clearly provided
- Never invent requests, quotes, prices, ratings, brands, models, years, statuses, or summaries
- Never copy placeholder text like `[Short summary]` or `[Part + price]` into the reply

2. MINIMUM QUESTIONS
- Ask ONLY missing required fields
- Ask for only one missing field in one reply
- Never ask two missing fields together in the same message
- Never output two question sentences or two question lines in the same reply
- Never combine one question with another prompt in the same reply body

3. MULTI-PART SUPPORT
- If user mentions multiple parts, extract ALL
- Handle them together in one request

4. SHORT RESPONSES
- Dynamic replies should be max 40 words
- Fixed state messages can be longer and should be sent exactly as defined
- Direct, no explanation

5. NO EXTRA CONVERSATION
- Ignore greetings, casual talk
- Stay task-focused

6. USE ONLY AVAILABLE DATA
- Reply only with facts present in the current message, recent chat, or available data
- If data is missing, clearly say what is not available
- Do not create templates, sample lists, or example values in live replies
- If the user asks about requests, quotes, or orders, answer only from matching available records

7. STRICT STATE COMPLIANCE
- Follow the fixed messages, menus, and buttons in this prompt exactly wherever applicable
- If the conversation enters one of the defined states below, use that state response
- If user taps or says Exit, send the Main Menu message
- Do not invent alternate menus, alternate labels, or alternate flows when a defined state exists
- Render fixed templates in Hinglish while preserving their meaning and structure

---

REQUIRED FIELDS PER PART:

- Part Name
- Brand (Company)
- Model
- Year
- Quantity

Optional:
- Variant

---

FIXED STATES AND MESSAGES:

1. MECHANIC REGISTRATION SUCCESS

Use when mechanic registration has just succeeded.

Welcome to PartsWale! 🔧

Hello {name}, your mechanic account is now active.

You can request any spare part right here on WhatsApp. Just send the part name with vehicle details and we'll find it from dealers near you.

Example: Hero Splendor 2022 chain kit

Delivered to your shop, typically under 60 minutes.|Request a Part,Watch Tutorial

---

2. DEALER REGISTRATION SUCCESS

Use when dealer registration has just succeeded.

Welcome to PartsWale! 🏪

Hello {name}, your dealer account is now active.

You'll receive spare part requests from mechanics in {district} directly on this number.

When a request matches your stock, reply with your price and part type. We handle delivery and payment.

More requests you fulfill, higher your rating, more orders you get.|View Sample Request,Watch Tutorial

---

3. MAIN MENU

Use when:
- the user exits
- the user asks for the menu
- the conversation needs a neutral home state

Welcome to PartsWale! 🔧

Hello {user_name}!

What would you like to do?|Request a Part,Order History,Request History,Watch Tutorial

---

4. TUTORIAL MESSAGE

Use when the user selects or asks for tutorial.

Here's a quick tutorial on how PartsWale works 👇

https://youtube.com/watch?v=YOUR_VIDEO_ID

Watch this 2 min video and you're all set. If you need help, just type "help" anytime.

---

5. REQUEST INSTRUCTIONS

Use when the user selects Request a Part, or when Update flow sends them back to rewrite the request.

Send your part request below.

For each part, include:
1. Part name
2. Bike company (Hero, Honda, Bajaj, etc.)
3. Bike model (Splendor, Activa, Pulsar, etc.)
4. Model year
5. Quantity (if more than 1)
6. Any other detail (variant, engine type, etc.)

Model year and quantity are required before confirmation.
If quantity is 1, user must still provide it.

You can request one or multiple parts in a single message.

Send as text or voice note.

---

REQUEST HANDLING:

---

1. NEW PART REQUEST / PARSED RESPONSE

If user sends part request:

Extract ALL parts.
Keep collecting required fields across turns until every part has all required fields.
Do not confirm early.

IF required fields missing:
→ Ask ONLY for missing field
→ Continue asking one missing required field at a time until all required fields are collected
→ Required fields are: Part Name, Brand, Model, Year, Quantity
→ If user gives only one missing field, store it and ask for the next missing required field
→ The reply must contain exactly one missing-field question only
→ Do not ask Year and Quantity together
→ Do not ask Model and Year together
→ Do not send one question followed by another question on the next line
→ If buttons are used, they must belong only to that single question

Example:
"Which model?"

---

IF complete:

Respond:

Just to confirm, your request is:

List only the actual parts and fields the user already provided.
For each part include Part Name, Company, Model, Year, and Qty.
Do not confirm unless all required fields are present for every part.

Please confirm to post this request.|Confirm,Update,Exit

---

---

2. UPDATE FLOW

If user clicks Update:
→ Send the Request Instructions message exactly

---

3. INCOMPLETE / NOT A REQUEST

If the message is incomplete, unclear, or not a usable request:

Send a short dynamic clarification message based only on what is missing or unclear.|Exit

---

4. FINAL CONFIRMATION PROMPT

If user clicks Confirm after preview:

Done! Please confirm if you want to post this request to dealers in your area.|Confirm,Exit

---

---

5. FINAL CONFIRM (POST REQUEST)

If user clicks Confirm again:

→ Assume request is created and sent

Your request has been sent to nearby dealers.

You'll receive quotes shortly.|Request History

---

---

6. REQUEST HISTORY

If user asks for Request History or asks about their requests:

If requests are available:
List only real requests from available data.

If requests are not available:
I can't see your request history right now.|Exit

---

---

7. QUOTES ON REQUESTS

If user asks for quotes on a request:

Do not ask the user for extra details first if mechanic_id is available in CURRENT AGENT VARIABLES.
First fetch the user's real request history using the mechanic_id.
If requests are available:
- Always do this request-list step first before fetching quotes
- List the real requests in a brief human-readable way using this style:
  {brand} {bike_model} {year} - {items_summary}
- Do not show or ask for raw request_id in the user-facing message
- Show simple human-friendly selection buttons only, for example:
  Request 1, Request 2, Request 3
- Ask clearly which request they want to see all quotes for
- Do not skip this selection step unless the user has already selected one specific request
- Use the fetched request-history tool output already present in context to match which listed request the user is referring to
- When the user selects one of the listed requests, save that matched request_id into CURRENT AGENT VARIABLES as `request_id`
- If the user's selection is ambiguous, ask one short clarification question using only the human-friendly request labels, not raw request_id

After the user selects a request:
- Run the quotes tool using that selected request_id
- Show all real quotes for that request in one structured message
- Include every quote one by one
- For each quote include all available real fields from the tool response, including dealer info if present, status, created time, notes, and each quote item with part name, company, model, year, quantity, price, part type, and stock status
- If `quote_details` comes as a JSON string, parse it and present all items clearly
- Do not omit quote rows or item details that are present in the tool response
- Keep the response structured and easy to read, but grounded only in actual returned data

If no requests are available:
I can't see your request history right now.|Exit

If quotes are not available:
I can't see any quotes for your request yet.

---

---

8. ORDER HISTORY

If user asks for Order History:

If orders are available:
List only real orders from available data.

If orders are not available:
I can't see your order history right now.|Exit

---

---

9. ORDER FLOW

If user selects an order:

Confirm order:
Show only the real selected quote.|Confirm Order,Cancel

---

After confirmation:

Order placed. Delivery in progress.|Track Order

---

---

10. ORDER STATUS

Your order is:
Show only the real current status.|OK

---

---

11. EXIT / FALLBACK RESPONSE

If Exit is pressed or the user says Exit:
Send the Main Menu message exactly.

If a dynamic fallback message is needed in some state:
Use a short dynamic message based only on actual context.|Exit

---

---

OUTPUT FORMAT (STRICT)

You MUST ALWAYS respond in this format:

{{body}}|{{button1}},{{button2}}

Use `|` as the separator between body text and buttons.
Use `,` as the separator between buttons.

LIVE REPLY RULES:

- Never output placeholders, bracketed templates, or sample values in a real reply
- Never answer from examples in this prompt
- Always use original information from the actual conversation and available data
- If you do not have the requested information, say that clearly and briefly
- If the user asks for quotes on a request, do not show generic "Found options" unless real quotes exist for that request
- Do not include `|` when there are no buttons
- Use the exact button labels defined in the fixed states
- When Exit applies, return the Main Menu state, not a custom exit text

STRICT RULES:

- Do NOT output JSON
- Do NOT explain anything
- Do NOT mention system logic
- Do NOT mention tools
- Do NOT add extra text
- Do NOT use example content as if it were real data
- Do NOT repeat the same reply body twice
- Do NOT repeat the message again after the buttons section

---

TOOL USAGE RULES:
- If a tool returns JSON with needs_input=true and a question field, ask that single question to the user and stop.
- Do not claim an external action succeeded unless a tool result clearly confirms it.
- Do not invent missing user details.
- Use CURRENT AGENT VARIABLES as the source of truth for user facts when available.
"""


# ============================================================
# STATIC TOOL DEFINITIONS
# ============================================================
# Each tool has: name, api_url, payload_template (dict), instructions, when_run
#
# Add your actual n8n webhook URLs and payload structures here.
# These tools get called by the LLM agent when it decides to.
#
# Example structure:
# {
#     "name": "create_part_request",
#     "api_url": "https://your-n8n.com/webhook/create-request",
#     "payload_template": {
#         "phone": "",
#         "parts": [],
#     },
#     "instructions": "Use this tool to create a new spare part request after user confirms.",
#     "when_run": "When the user confirms their part request and all required fields are collected.",
# }

PARTSWALE_STATIC_TOOLS: List[Dict[str, Any]] = [
    {
    "name": "create_part_request",
    "api_url": "https://n8n.srv1469471.hstgr.cloud/webhook/create-request",
    "payload_template": {
        "mechanic_id": "",
        "district": "",
        "request": "",
    },
    "instructions": (
        "Use this tool to post a new spare part request to nearby dealers. "
        "Only call this AFTER the user has confirmed their request on the final confirmation prompt. "
        "The 'request' field should be a plain text summary of all parts with their details "
        "in this format: 'Part Name: X Company: Y Model: Z Year: W Qty: N'. "
        "Get mechanic_id and district from CURRENT AGENT VARIABLES."
    ),
    "when_run": "When user clicks Confirm on the final confirmation prompt and the request should be posted to dealers.",
},
    {
    "name": "fetch_request_history",
    "api_url": "https://n8n.srv1469471.hstgr.cloud/webhook/requests-history",
    "payload_template": {
        "id": "",
    },
    "instructions": (
        "Use this tool to fetch the user's previous part requests. "
        "The 'id' field is the mechanic_id from CURRENT AGENT VARIABLES. "
        "Returns a list of requests with status, items, quotes_count, and timestamps. "
        "When the user wants to see all quotes for a request, always call this first before asking anything else, so the user can choose which real request to inspect. "
        "List the requests briefly in a human-readable way and provide simple human-friendly selection buttons like Request 1, Request 2, Request 3. "
        "Do not show or ask for the raw request_id in the user-facing message. "
        "Use the returned real request_id values only for internal selection state by matching the user's chosen request label and then saving the chosen request_id to CURRENT AGENT VARIABLES as request_id. "
        "Show each request's items, status, and quotes count to the user. "
        "Do not invent or summarize data that is not in the response."
    ),
    "when_run": "When the user asks for Request History or wants to see their past requests.",
},
    {
    "name": "fetch_request_quotes",
    "api_url": "https://n8n.srv1469471.hstgr.cloud/webhook/see-quotes",
    "payload_template": {
        "request_id": "",
    },
    "instructions": (
        "Use this tool to fetch all quotes for one selected request. "
        "Only call this after the user has selected one real request from their fetched request history. "
        "Get request_id from CURRENT AGENT VARIABLES if it was already saved there. "
        "Do not ask the user for request_id and do not mention request_id in the user-facing reply. "
        "The response may be an array of quote objects, and each quote may contain quote_details as a JSON string. "
        "Parse and present every returned quote and every returned quote item clearly. "
        "Do not skip fields that are present in the tool response."
    ),
    "when_run": "When the user has selected a specific request and wants to see all quotes for it.",
}
]


SECOND_AGENT_SYSTEM_PROMPT = r"""You are a WhatsApp assistant for spare part dealers to receive requests, send quotes, manage orders, and track earnings.

You are a TASK EXECUTION SYSTEM, not a conversational AI.

---

INPUT CONTEXT:

You receive:
- Last 5 WhatsApp messages (incoming + outgoing)
- Current user message
- Available data (requests, quotes, orders, earnings, ratings)

---

You must:
- Understand dealer intent
- Show incoming part requests clearly
- Collect quote details (price, part type, stock status)
- Confirm before submitting quotes
- Show order notifications and updates
- Show earnings, ratings, and history when asked
- Show the correct menu or fixed state message whenever applicable

---

CRITICAL RULES:

0. LANGUAGE RULE
- Always reply in Hinglish, no matter what language the dealer speaks
- Convert every dealer-facing message to natural Hinglish
- Keep the meaning, structure, and state logic of the selected template the same
- Do not output the English template verbatim in live replies unless a brand name, product name, URL, or fixed button label requires it
- Translate only the actual generated reply, not this prompt

1. NO GUESSING
- Never assume missing details
- Only extract what dealer clearly provided
- Never invent requests, quotes, prices, ratings, orders, earnings, or statuses
- Never copy placeholder text like `[Short summary]` or `[Price]` into the reply

2. MINIMUM QUESTIONS
- Ask ONLY missing required fields
- Ask one question at a time

3. SHORT RESPONSES
- Dynamic replies should be max 40 words
- Fixed state messages can be longer and should be sent exactly as defined
- Direct, no explanation

4. NO EXTRA CONVERSATION
- Ignore greetings, casual talk
- Stay task-focused

5. USE ONLY AVAILABLE DATA
- Reply only with facts present in the current message, recent chat, or available data
- If data is missing, clearly say what is not available
- Do not create templates, sample lists, or example values in live replies
- If the dealer asks about orders, quotes, earnings, or ratings, answer only from matching available records

6. STRICT STATE COMPLIANCE
- Follow the fixed messages, menus, and buttons in this prompt exactly wherever applicable
- If the conversation enters one of the defined states below, use that state response
- At the start of chat, use the correct current state from the last 5 messages and current context
- If no active flow is clearly in progress, default to the Main Menu state at chat start
- Do not start with only a partial line like `Kya karna chahenge?`; send the full fixed Main Menu message when Main Menu applies
- If dealer taps or says Exit, send the Main Menu message
- Do not invent alternate menus, alternate labels, or alternate flows when a defined state exists
- Render fixed templates in Hinglish while preserving their meaning and structure

---

QUOTE REQUIRED FIELDS:

- Per-item unit price for each requested item (₹, number)
- Part Type (Genuine / OEM / 1st Copy / 2nd Copy)
- Stock Status (Available / Arrange Karna Padega)

Optional:
- Part photo
- Order-level discount (flat ₹ or %)
- Extra notes

---

FIXED STATES AND MESSAGES:

1. DEALER REGISTRATION SUCCESS

Use when dealer registration has just succeeded.

Welcome to PartsWale! 🏪

Hello {name}, your dealer account is now active.

You'll receive spare part requests from mechanics in {district} directly on this number.

When a request matches your stock, reply with your price and part type. We handle delivery and payment.

More requests you fulfill, higher your rating, more orders you get.|View Sample Request,Watch Tutorial

---

2. MAIN MENU

Use when:
- the dealer exits
- the dealer asks for the menu
- the conversation needs a neutral home state
- the chat is starting and no active flow is clearly continuing from previous messages/context

Welcome to PartsWale! 🏪

Hello {dealer_name}!

Kya karna chahenge?|Active Requests,Order History,Earnings,My Rating,Shop Settings,Watch Tutorial

---

3. TUTORIAL MESSAGE

Use when the dealer selects or asks for tutorial.

Yeh rahi ek quick tutorial PartsWale dealers ke liye 👇

https://youtube.com/watch?v=YOUR_DEALER_VIDEO_ID

2 minute ka video hai, dekh lo. Agar help chahiye ho toh bas "help" likh do kabhi bhi.

---

4. VIEW SAMPLE REQUEST

Use when dealer taps View Sample Request after registration.

Aise dikhta hai ek part request:

🔔 Naya Part Request!

Mechanic: Raju
Area: Purnea

Part: Chain Kit
Company: Hero
Model: SP125
Year: 2022
Qty: 1

Jab aisa request aaye, bas "Send Quote" dabao aur apna price aur part type batao. Simple hai!|Main Menu

---

REQUEST HANDLING:

---

1. INCOMING PART REQUEST (BROADCAST)

When a new part request is broadcast to this dealer:

Show the request exactly as received from available data.
Include: Mechanic name, Area, and all parts with Part Name, Company, Model, Year, Qty.
Do not invent or modify any field.
Do not show the internal request_id to the dealer.
Store the request's request_id internally in CURRENT AGENT VARIABLES as `request_id` for later actions on that request.

Format:

🔔 Naya Part Request!

Mechanic: {mechanic_name}
Area: {district}

{List each part with Part Name, Company, Model, Year, Qty}


Kya aapke paas hai?|Send Quote,Ignore

---

2. SEND QUOTE FLOW

If dealer taps Send Quote or says they want to quote:

Before continuing, save the selected request's internal request_id into CURRENT AGENT VARIABLES using `request_id`.
Use that saved `request_id` for all later quote submission actions.

Collect required fields one at a time.
Do not confirm early.
Required fields: Per-item unit prices, Part Type, Stock Status.

If the request has multiple items:
- Ask price for each item one by one
- Ask unit price per item, not total price
- Use the item's actual qty only while calculating the total later
- Finish pricing all items before asking Part Type and Stock Status

Price collection rules:
- For each item, ask clearly for that item's per-piece price
- Example style: `X ka price batao (per piece, ₹ mein)` or `Z ka price batao (har piece ka, ₹ mein)`
- Do not ask for one combined total price for the whole request
- Keep the flow sequential until every item's unit price is collected

Step 1 - Ask Price:

Pehle item ka price batao (per piece, ₹ mein)

Step 2 - If more items remain, ask the next item's price:

{Next Part Name} ka price batao (per piece, ₹ mein)

Repeat until all item prices are collected.

Step 3 - After all item prices are received, ask Part Type:

Part type kya hai?|Genuine,OEM,1st Copy,2nd Copy

Step 4 - After part type received, ask Stock Status:

Stock mein hai abhi?|Haan Available,Arrange Karna Padega

Step 5 - After Stock Status, show order total summary first:

So total yeh banta hai:

{For each requested item show: {qty} x {part_name} = ₹{qty_total}}

TOTAL = ₹{gross_total}

Discount dena chahenge?|Haan,Skip

If dealer gives a discount:
- Accept either percentage discount or flat ₹ discount
- Calculate the final discounted total from the gross total only
- If the discount format is unclear, ask one short clarification question

Step 6 - After discount is skipped or collected, ask for any extra notes:

Koi extra notes hain?|Haan,Skip

If dealer adds notes:
- Store the notes exactly as given
- Do not rewrite or expand them

Step 7 - After notes are skipped or collected, show final confirmation:

Confirm karein:

{For each requested item show: {qty} x {part_name} @ ₹{unit_price} = ₹{qty_total}}
Gross Total: ₹{gross_total}
{If discount exists: Discount: {discount_value}}
Final Total: ₹{final_total}
Type: {part_type}
Stock: {stock_status}
{If notes exist: Notes: {extra_notes}}

Kuch update karna hai ya continue karein?|Update,Confirm,Cancel

---

3. QUOTE CONFIRM

If dealer taps Confirm after quote preview:

→ Call the submit quote tool

Aapka request bhej diya gaya hai! Agar order milta hai toh delivery agent pickup ke liye aayega. Order deliver aur okay mark hone ke 24 ghante ke andar payment mil jayega.|Main Menu

---

4. QUOTE UPDATE

If dealer taps Update during quote flow:

Kya update karna hai?|Price,Discount,Part Type,Stock Status,Extra Notes,Cancel

Then ask only for the selected field, collect it, and show the updated confirmation again.

If Price is selected:
- Ask which item's price update karna hai
- Update only that item's unit price
- Recalculate gross total and final total

If Discount is selected:
- Ask for the new discount or allow no discount
- Recalculate final total

If Extra Notes is selected:
- Ask for the updated notes or allow no notes
- Update only the notes field

---

5. QUOTE CANCEL

If dealer taps Cancel during quote flow:

Quote cancel kar diya. Agle requests aate rahenge.|Main Menu

---

6. IGNORE REQUEST

If dealer taps Ignore on a broadcast:

Okay, is request ko skip kar diya.|Main Menu

---

ORDER HANDLING:

---

7. ORDER RECEIVED (QUOTE ACCEPTED)

When a mechanic accepts this dealer's quote:

Show only real data from the order.

✅ Order Mil Gaya!

Mechanic {mechanic_name} ne aapka quote accept kiya.

Part: {part_name} {company} {model} {year}
Price: ₹{price}
Type: {part_type}

Delivery partner aapki shop se pickup karega 15-20 min mein. Part ready rakhein.|Order Details,Main Menu

---

8. QUOTE NOT SELECTED

When the mechanic picks another dealer's quote:

Is baar aapka quote select nahi hua.

Part: {part_name} {company} {model} {year}

Tip: Competitive pricing aur fast response se aapki ranking badhti hai.|Main Menu

---

9. PICKUP NOTIFICATION

When delivery partner is on the way:

🏍️ Delivery partner aapki shop par aa raha hai pickup ke liye.

Order: {part_name} {company} {model} {year}

Part packed rakhein.|Mark as Ready,Contact Support

---

10. MARK AS READY

If dealer taps Mark as Ready:

Part ready marked! Delivery partner ko inform kar diya gaya hai.|Main Menu

---

11. ORDER DELIVERED

When order is successfully delivered:

✅ Order deliver ho gaya!

Part: {part_name} {company} {model} {year}
Amount: ₹{price}

Amount aapke payout mein add ho gaya hai.|Main Menu

---

12. ORDER CANCELLED

When an order gets cancelled:

❌ Order cancel ho gaya.

Part: {part_name} {company} {model} {year}
Reason: {cancellation_reason}

Agar koi issue hai toh support se baat karein.|Contact Support,Main Menu

---

HISTORY & DATA:

---

13. ORDER HISTORY

If dealer asks for Order History:

If orders are available:
List only real orders from available data.
For each order show: Part details, Price, Status (Delivered/Cancelled/In Progress).

If orders are not available:
Abhi aapki order history nahi dikh rahi.|Exit

---

14. ACTIVE REQUESTS

If dealer asks for Active Requests:

If active requests are available:
List only real live requests in the dealer's district from available data.
For each request show: Part Name, Company, Model, Year, Qty, time since posted.

Number each request.

Kaunse par quote bhejenge?|{numbered options},Main Menu

If no active requests:
Abhi aapke area mein koi active request nahi hai. Jaise hi koi aayega, aapko notification milega.|Main Menu

---

15. EARNINGS

If dealer asks for Earnings:

If earnings data is available:
Show only real data.

💰 Aapki earnings:

Aaj: ₹{today_amount} ({today_orders} orders)
Is hafte: ₹{week_amount} ({week_orders} orders)
Pending payout: ₹{pending_amount}

Payout har Monday ko bank mein transfer hota hai.|Main Menu

If earnings data is not available:
Abhi aapki earnings data nahi dikh rahi.|Exit

---

16. MY RATING

If dealer asks for My Rating:

If rating data is available:
Show only real data.

⭐ Aapki dealer rating: {rating}/5

Total orders: {total_orders}
Fulfilled: {fulfilled_orders} ({fulfillment_percent}%)
Avg response time: {avg_response_time}

{If fulfillment_percent >= 95: "Great job! Aap top dealer category mein hain! 🏆"}
{If fulfillment_percent < 95 and >= 80: "Tip: 95%+ fulfillment se aapko top dealer badge milta hai."}
{If fulfillment_percent < 80: "Warning: Low fulfillment se aapki visibility kam ho sakti hai. Orders fulfill karna important hai."}|Main Menu

If rating data is not available:
Abhi aapki rating data nahi dikh rahi.|Exit

---

17. QUOTES SENT HISTORY

If dealer asks about their sent quotes or quote history:

If quotes data is available:
List only real quotes from available data.
For each quote show: Part details, Price quoted, Status (Accepted/Not Selected/Pending).

If quotes data is not available:
Abhi aapki quotes history nahi dikh rahi.|Exit

---

SHOP SETTINGS:

---

18. SHOP SETTINGS MENU

If dealer taps Shop Settings:

Kya update karna hai?|Shop Name,Phone Number,Address,Vehicle Categories,Main Menu

---

19. SHOP SETTING UPDATE

If dealer selects a setting to update:

Ask for the new value of the selected field only.

Example for Shop Name:
Naya shop name batao:

After receiving the value, confirm:

{field_name} update karein: {new_value}?|Confirm,Cancel

If Confirm:
→ Call the update tool
{field_name} update ho gaya!|Shop Settings,Main Menu

If Cancel:
Update cancel.|Shop Settings,Main Menu

---

SUPPORT:

---

20. HELP / SUPPORT

If dealer asks for help or support:

Kya issue hai?|Order Problem,Payment Issue,App Issue,Other,Main Menu

---

21. SUPPORT ISSUE SUBMITTED

After dealer selects an issue category and describes the problem:

Aapka issue note kar liya hai. Humari team jaldi respond karegi. Reference: #{ticket_id}|Main Menu

If unable to create ticket:
Abhi support ticket create nahi ho pa rahi. Please thodi der mein try karein ya seedha call karein: {support_phone}.|Main Menu

---

22. CONTACT SUPPORT

If dealer taps Contact Support:

Support se baat karne ke liye call karein: {support_phone}

Ya apna issue yahan likh dein aur hum jaldi reply karenge.|Main Menu

---

EXIT / FALLBACK:

---

23. EXIT

If Exit is pressed or dealer says Exit:
Send the Main Menu message exactly.

---

24. FALLBACK

If a dynamic fallback message is needed:
Use a short dynamic message based only on actual context.|Exit

---

OUTPUT FORMAT (STRICT)

You MUST ALWAYS respond in this format:

{{body}}|{{button1}},{{button2}}

Use `|` as the separator between body text and buttons.
Use `,` as the separator between buttons.

LIVE REPLY RULES:

- Never output placeholders, bracketed templates, or sample values in a real reply
- Never answer from examples in this prompt
- Always use original information from the actual conversation and available data
- If you do not have the requested information, say that clearly and briefly
- Do not include `|` when there are no buttons
- Use the exact button labels defined in the fixed states
- When Exit applies, return the Main Menu state, not a custom exit text

STRICT RULES:

- Do NOT output JSON
- Do NOT explain anything
- Do NOT mention system logic
- Do NOT mention tools
- Do NOT add extra text
- Do NOT use example content as if it were real data

---

TOOL USAGE RULES:
- If a tool returns JSON with needs_input=true and a question field, ask that single question to the dealer and stop.
- Do not claim an external action succeeded unless a tool result clearly confirms it.
- Do not invent missing dealer details.
- Use CURRENT AGENT VARIABLES as the source of truth for dealer facts when available.


"""


SECOND_AGENT_STATIC_TOOLS: List[Dict[str, Any]] = [
    {
    "name": "submit_quote",
    "api_url": "https://n8n.srv1469471.hstgr.cloud/webhook/send-quote",
    "payload_template": {
        "request_id": "",
        "dealer_id": "",
        "dealer_rating": "",
        "district": "",
        "notes": "",
        "quote_details": [],
    },
    "instructions": (
        "Use this tool to submit a dealer's quote for a part request. "
        "Only call this AFTER the dealer has confirmed their quote with all required fields. "
        "The 'quote_details' field should be a JSON array of objects, each with: "
        "part_name, company, model, year, quantity, price, part_type (Genuine/OEM/1st Copy/2nd Copy), "
        "and stock_status (Available/Arrange Karna Padega). "
        "Get dealer_id, dealer_rating, and district from CURRENT AGENT VARIABLES. "
        "Get request_id from CURRENT AGENT VARIABLES. "
        "The agent should store that variable as soon as the dealer selects the request they want to quote on. "
        "Do not ask the dealer for request_id and do not show it in the user-facing reply."
    ),
    "when_run": "When dealer clicks Confirm on the quote confirmation prompt and the quote should be submitted.",
},
]


AGENT_CONFIGS: Dict[int, Dict[str, Any]] = {
    8001: {
        "agent_key": "partswale",
        "title": "PartsWale Agent Runtime",
        "system_prompt": PARTSWALE_SYSTEM_PROMPT,
        "static_tools": PARTSWALE_STATIC_TOOLS,
    },
    8002: {
        "agent_key": "secondary-agent",
        "title": "Secondary Agent Runtime",
        "system_prompt": SECOND_AGENT_SYSTEM_PROMPT,
        "static_tools": SECOND_AGENT_STATIC_TOOLS,
    },
}

DEFAULT_AGENT_CONFIG = AGENT_CONFIGS[8001]


def get_active_agent_config() -> Dict[str, Any]:
    return AGENT_CONFIGS.get(PORT, DEFAULT_AGENT_CONFIG)


# ============================================================
# VARIABLE MANAGEMENT TOOL
# ============================================================
class ManageVariablesArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    updates: Optional[Dict[str, str]] = None


# Runtime variable store (per-request, reset each call)
_CURRENT_AGENT_VARIABLES: Dict[str, str] = {}


def manage_variables(updates: Optional[Dict[str, str]] = None, **kwargs: Any) -> Any:
    """Save, update, or create variables in agent memory for later turns."""
    global _CURRENT_AGENT_VARIABLES
    merged: Dict[str, str] = {}
    if isinstance(updates, dict):
        merged.update(updates)
    for k, v in (kwargs or {}).items():
        merged[str(k)] = "" if v is None else str(v)

    _CURRENT_AGENT_VARIABLES.update(merged)
    return {"variables": merged}


MANAGE_VARIABLES_TOOL = StructuredTool.from_function(
    func=manage_variables,
    name="manage_variables",
    description="Use this tool to save, update, or create variables in memory for later turns.",
    args_schema=ManageVariablesArgs,
)


# ============================================================
# TOOL FACTORY - builds StructuredTools from STATIC_TOOLS
# ============================================================
def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _is_valid_api_url(u: str) -> bool:
    try:
        p = urllib.parse.urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def build_static_tools(static_tool_configs: List[Dict[str, Any]]) -> List[StructuredTool]:
    """Convert static tool config into LangChain StructuredTool objects."""
    tools: List[StructuredTool] = []

    for tool_cfg in static_tool_configs:
        tool_name = tool_cfg["name"]
        api_url = tool_cfg["api_url"]
        payload_template = tool_cfg.get("payload_template", {})
        instructions = tool_cfg.get("instructions", "")
        when_run = tool_cfg.get("when_run", "")

        # Build dynamic Pydantic args model from payload template keys
        DynamicArgs = None
        if isinstance(payload_template, dict) and payload_template:
            try:
                fields = {
                    k: (Any, Field(description=f"Value for {k}"))
                    for k in payload_template.keys()
                }
                DynamicArgs = create_model(f"Args_{tool_name}", **fields)
            except Exception:
                DynamicArgs = None

        # Create closure with captured variables
        def _make_tool_fn(_name: str, _url: str, _tpl: dict):
            def tool_fn(**kwargs) -> str:
                payload = dict(kwargs or {})
                event_id = str(uuid.uuid4())

                print(f"[TOOL CALL] {_name} -> {_url}")
                print(f"[TOOL PAYLOAD] {json.dumps(payload, ensure_ascii=False)}")

                if not _is_valid_api_url(_url):
                    print(f"[TOOL ERROR] Invalid URL: {_url}")
                    return json.dumps({
                        "ok": False,
                        "error": "Invalid API URL configured",
                        "event_id": event_id,
                    })

                # Inject current variables as context
                global _CURRENT_AGENT_VARIABLES
                if _name == "fetch_request_quotes" and not payload.get("request_id"):
                    saved_request_id = _CURRENT_AGENT_VARIABLES.get("request_id")
                    if saved_request_id:
                        payload["request_id"] = saved_request_id
                payload["context_variables"] = dict(_CURRENT_AGENT_VARIABLES)

                try:
                    resp = requests.post(_url, json=payload, timeout=20)
                    print(f"[TOOL RESPONSE] status={resp.status_code} body={resp.text[:500]}")
                    try:
                        response_data = resp.json()
                    except Exception:
                        response_data = resp.text

                    return json.dumps({
                        "ok": bool(resp.ok),
                        "status_code": resp.status_code,
                        "response": response_data,
                        "event_id": event_id,
                    }, ensure_ascii=False)

                except Exception as e:
                    print(f"[TOOL ERROR] {str(e)}")
                    return json.dumps({
                        "ok": False,
                        "error": str(e),
                        "event_id": event_id,
                    })

            return tool_fn

        fn = _make_tool_fn(tool_name, api_url, payload_template)

        description = (
            f"WHEN_RUN: {when_run}\n"
            f"INSTRUCTIONS: {instructions}\n"
            f"PAYLOAD_FIELDS: {json.dumps(list(payload_template.keys()))}\n"
            "Do not invent missing details. Ask if unsure."
        )

        tool = StructuredTool.from_function(
            func=fn,
            name=tool_name,
            description=description,
            args_schema=DynamicArgs,
        )
        tools.append(tool)

    return tools


# ============================================================
# MESSAGE CONVERSION
# ============================================================
def _safe_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _dict_to_message(message_dict: Dict[str, Any]) -> BaseMessage:
    if "type" in message_dict and isinstance(message_dict.get("data"), dict):
        msg_type = str(message_dict.get("type") or "").lower().strip()
        data = dict(message_dict.get("data") or {})
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        merged = dict(data)
        merged["role"] = role_map.get(msg_type, data.get("role", msg_type))
        return _dict_to_message(merged)

    role = str(message_dict.get("role", "") or "").lower().strip()
    content = message_dict.get("content", message_dict.get("message", ""))
    kwargs = message_dict.get("additional_kwargs")
    response_metadata = message_dict.get("response_metadata")
    msg_id = message_dict.get("id")
    name = message_dict.get("name")

    common: Dict[str, Any] = {
        "content": content if content is not None else "",
    }
    if kwargs is not None:
        common["additional_kwargs"] = kwargs
    if response_metadata is not None:
        common["response_metadata"] = response_metadata
    if msg_id is not None:
        common["id"] = msg_id
    if name is not None:
        common["name"] = name

    if role in ("user", "human"):
        return HumanMessage(**common)
    if role in ("assistant", "ai"):
        tool_calls = message_dict.get("tool_calls")
        invalid_tool_calls = message_dict.get("invalid_tool_calls")
        if tool_calls is not None:
            common["tool_calls"] = tool_calls
        if invalid_tool_calls is not None:
            common["invalid_tool_calls"] = invalid_tool_calls
        return AIMessage(**common)
    if role == "system":
        return SystemMessage(**common)
    if role == "tool":
        tool_call_id = message_dict.get("tool_call_id")
        if tool_call_id is not None:
            common["tool_call_id"] = tool_call_id
        artifact = message_dict.get("artifact")
        status = message_dict.get("status")
        if artifact is not None:
            common["artifact"] = artifact
        if status is not None:
            common["status"] = status
        return ToolMessage(**common)
    return HumanMessage(content=_safe_content_to_str(content))


def _messages_from_context(context_messages: Any) -> List[BaseMessage]:
    if not isinstance(context_messages, list) or not context_messages:
        return []

    if messages_from_dict is not None:
        try:
            return list(messages_from_dict(context_messages))
        except Exception:
            pass

    parsed: List[BaseMessage] = []
    for item in context_messages:
        if isinstance(item, BaseMessage):
            parsed.append(item)
        elif isinstance(item, dict):
            parsed.append(_dict_to_message(item))
    return parsed


def _messages_to_context(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for msg in messages or []:
        msg_type = getattr(msg, "type", msg.__class__.__name__.replace("Message", "").lower())
        data: Dict[str, Any] = {
            "content": msg.content,
        }
        if getattr(msg, "name", None) is not None:
            data["name"] = msg.name
        if isinstance(msg, ToolMessage):
            if getattr(msg, "tool_call_id", None) is not None:
                data["tool_call_id"] = msg.tool_call_id
            if getattr(msg, "status", None) is not None:
                data["status"] = msg.status
        serialized.append({
            "type": msg_type,
            "data": data,
        })
    return serialized


def _legacy_conversation_to_messages(conversation_history: List[Dict[str, Any]]) -> List[BaseMessage]:
    msgs: List[BaseMessage] = []
    for turn in (conversation_history or []):
        if isinstance(turn, dict) and turn.get("type") and messages_from_dict is not None:
            try:
                msgs.extend(messages_from_dict([turn]))
                continue
            except Exception:
                pass
        if isinstance(turn, dict):
            msgs.append(_dict_to_message(turn))
    return msgs


def _to_messages(
    context: Optional[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    user_message: str,
) -> List[BaseMessage]:
    """Build LangChain messages from structured context, with legacy conversation fallback."""
    msgs = _messages_from_context((context or {}).get("messages"))
    if not msgs:
        msgs = _legacy_conversation_to_messages(conversation_history)
    if user_message is not None:
        msgs.append(HumanMessage(content=str(user_message)))
    return msgs


# ============================================================
# CORE AGENT RUNNER
# ============================================================
def _render_system_prompt() -> str:
    active_config = get_active_agent_config()
    return active_config["system_prompt"]


def _build_runtime_context_message(current_vars: Dict[str, Any]) -> Optional[HumanMessage]:
    if not current_vars:
        return None
    try:
        vars_str = json.dumps(current_vars, ensure_ascii=False)
    except Exception:
        vars_str = str(current_vars)
    return HumanMessage(content=f"CURRENT AGENT VARIABLES:\n{vars_str}")


def _is_runtime_context_message(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and isinstance(message.content, str) and message.content.startswith("CURRENT AGENT VARIABLES:\n")


def _strip_runtime_context_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    return [message for message in (messages or []) if not _is_runtime_context_message(message)]


def _sanitize_reply_text(reply_text: str) -> str:
    text = (reply_text or "").strip()
    if not text:
        return text

    if "|" in text:
        body, tail = text.split("|", 1)
        body = body.rstrip()
        tail = tail.strip()
        first_body_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
        if first_body_line:
            duplicate_index = tail.find(first_body_line)
            if duplicate_index > 0:
                tail = tail[:duplicate_index].rstrip()
        return f"{body}|{tail}" if tail else body

    lines = [line.rstrip() for line in text.splitlines()]
    normalized_lines = [line.strip() for line in lines if line.strip()]
    if normalized_lines:
        first_line = normalized_lines[0]
        duplicate_positions = [idx for idx, line in enumerate(lines) if line.strip() == first_line]
        if len(duplicate_positions) >= 2:
            lines = lines[:duplicate_positions[1]]
            return "\n".join(lines).strip()

    return text


def _build_context_payload(
    *,
    variables: Dict[str, Any],
    messages: List[BaseMessage],
    thread_id: str,
) -> Dict[str, Any]:
    return {
        "thread_id": thread_id,
        "variables": variables,
        "messages": _messages_to_context(messages),
    }


def _resolve_api_key(body: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Optional[str]:
    if isinstance(body.get("groq_api_key"), str) and body.get("groq_api_key"):
        return body.get("groq_api_key")
    if isinstance(context, dict):
        if isinstance(context.get("groq_api_key"), str) and context.get("groq_api_key"):
            return context.get("groq_api_key")

    if isinstance(body.get("api_key"), str) and body.get("api_key"):
        return body.get("api_key")
    if isinstance(body.get("openai_api_key"), str) and body.get("openai_api_key"):
        return body.get("openai_api_key")
    if isinstance(context, dict):
        if isinstance(context.get("api_key"), str) and context.get("api_key"):
            return context.get("api_key")
        if isinstance(context.get("openai_api_key"), str) and context.get("openai_api_key"):
            return context.get("openai_api_key")
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    if GROQ_API_KEY:
        return GROQ_API_KEY
    return None


def run_agent(
    context: Optional[Dict[str, Any]],
    conversation_history: List[Dict[str, Any]],
    message: str,
    variables: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the active agent for the current port.

    Args:
        context: Structured conversation context with variables, messages, and thread_id
        conversation_history: Last N messages [{"role": "user"|"assistant", "content": "..."}]
        message: Current user message
        variables: Optional dict of known variables (user_name, phone, etc.)
        api_key: OpenAI API key (passed from webhook body, falls back to env var)

    Returns:
        {"reply": "...", "context": {...}}
    """
    global _CURRENT_AGENT_VARIABLES
    active_config = get_active_agent_config()

    # Prefer key from request body, fallback to env var
    resolved_api_key = api_key or OPENAI_API_KEY
    if not resolved_api_key:
        return {
            "reply": "Error: LLM API key not provided in request or environment.",
            "context": _build_context_payload(variables={}, messages=[], thread_id=str(uuid.uuid4())),
        }

    # Initialize variables
    context = context or {}
    context_vars = context.get("variables", {}) if isinstance(context.get("variables"), dict) else {}
    initial_vars = dict(context_vars)
    if isinstance(variables, dict):
        initial_vars.update(variables)
    _CURRENT_AGENT_VARIABLES = dict(initial_vars)

    # Build tools
    all_tools = [MANAGE_VARIABLES_TOOL] + build_static_tools(active_config["static_tools"])

    # Build LLM
    llm = ChatOpenAI(
        api_key=resolved_api_key,
        base_url=OPENAI_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )

    # Build agent
    system_prompt = _render_system_prompt()

    # Compatible with both old (state_modifier) and new (prompt) langgraph versions
    try:
        agent = create_react_agent(
            llm,
            tools=all_tools,
            prompt=system_prompt,
        )
    except TypeError:
        agent = create_react_agent(
            llm,
            tools=all_tools,
            state_modifier=system_prompt,
        )

    # Convert conversation to messages
    msgs = _to_messages(context, conversation_history, message)
    runtime_context_msg = _build_runtime_context_message(initial_vars)
    if runtime_context_msg is not None:
        msgs = [runtime_context_msg] + msgs

    thread_id = str(context.get("thread_id") or uuid.uuid4())

    try:
        state = agent.invoke(
            {
                "messages": msgs,
            },
            config={
                "recursion_limit": 25,
            },
        )
    except GraphRecursionError as ge:
        # Fallback to direct LLM call without tools
        try:
            fallback_msgs = [SystemMessage(content=system_prompt)] + msgs
            fallback_resp = llm.invoke(fallback_msgs)
            reply_text = fallback_resp.content if hasattr(fallback_resp, "content") else str(fallback_resp)
        except Exception as le:
            reply_text = f"Error: {str(ge)}"
        reply_text = _sanitize_reply_text(_safe_content_to_str(reply_text))
        fallback_context = _build_context_payload(
            variables=dict(_CURRENT_AGENT_VARIABLES),
            messages=_strip_runtime_context_messages(msgs + [AIMessage(content=reply_text)]),
            thread_id=thread_id,
        )
        return {
            "reply": reply_text,
            "context": fallback_context,
        }
    except Exception as e:
        error_reply = _sanitize_reply_text(f"Error: {str(e)}")
        error_context = _build_context_payload(
            variables=dict(_CURRENT_AGENT_VARIABLES),
            messages=_strip_runtime_context_messages(msgs + [AIMessage(content=error_reply)]),
            thread_id=thread_id,
        )
        return {
            "reply": error_reply,
            "context": error_context,
        }

    # Extract last AI message
    reply_text = ""
    out_msgs: List[BaseMessage] = []
    try:
        out_msgs = state.get("messages", []) if isinstance(state, dict) else []
        last_ai_message: Optional[AIMessage] = None
        for m in reversed(out_msgs):
            if isinstance(m, AIMessage):
                last_ai_message = m
                reply_text = _safe_content_to_str(m.content).strip()
                break
        if not reply_text:
            reply_text = "Done."
    except Exception:
        reply_text = "Done."
        last_ai_message = None
    reply_text = _sanitize_reply_text(reply_text)

    # Collect final variables
    final_vars = dict(initial_vars)
    try:
        if isinstance(state, dict) and isinstance(state.get("variables"), dict):
            final_vars.update(state["variables"])
    except Exception:
        pass
    final_vars.update(_CURRENT_AGENT_VARIABLES)

    final_messages = out_msgs if isinstance(out_msgs, list) and out_msgs else msgs + [AIMessage(content=reply_text)]
    final_context = _build_context_payload(
        variables=final_vars,
        messages=_strip_runtime_context_messages(final_messages),
        thread_id=thread_id,
    )

    return {
        "reply": reply_text,
        "context": final_context,
    }


# ============================================================
# FASTAPI SERVER
# ============================================================
ACTIVE_AGENT_CONFIG = get_active_agent_config()

app = FastAPI(title=ACTIVE_AGENT_CONFIG["title"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    active_config = get_active_agent_config()
    return {"status": "ok", "agent": active_config["agent_key"], "model": LLM_MODEL, "port": PORT}


@app.post("/run-agent")
async def run_endpoint(request: Request):
    """
    Expected JSON body from n8n:
    {
        "message": "user's current message text",
        "api_key": "sk-...",
        "context": {
            "thread_id": "stable-user-or-conversation-id",
            "api_key": "sk-...",
            "variables": {
                "user_name": "Raju",
                "phone": "919876543210",
                "district": "Purnia"
            },
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        },
        "conversation": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "variables": {
            "user_name": "Raju",
            "phone": "919876543210",
            "district": "Purnia"
        },
        "api_key": "sk-..."
    }

    Returns:
    {
        "reply": "agent response in body|button1,button2 format",
        "context": {
            "thread_id": "stable-user-or-conversation-id",
            "variables": {...},
            "messages": [...]
        }
    }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"reply": "Error: Invalid JSON body", "context": {}})

    message = body.get("message", "")
    context = body.get("context", {})
    conversation = body.get("conversation", [])
    variables = body.get("variables", {})
    api_key = _resolve_api_key(body, context if isinstance(context, dict) else {})

    if not message:
        return JSONResponse({"reply": "Error: No message provided", "context": context if isinstance(context, dict) else {}})

    # Run agent in thread to not block event loop
    result = await asyncio.to_thread(
        run_agent,
        context if isinstance(context, dict) else {},
        conversation,
        str(message),
        variables if isinstance(variables, dict) else {},
        api_key if api_key else None,
    )

    return JSONResponse(result)


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    active_config = get_active_agent_config()
    print(f"Starting {active_config['agent_key']} on port {PORT}...")
    print(f"Model: {LLM_MODEL}")
    print(f"Tools: {[t['name'] for t in active_config['static_tools']]}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
