# from flask import request, jsonify
# from datetime import datetime, timedelta
# from helper.invoice_generator import generate_invoice_pdf
# import pymysql
# import json
# import razorpay


# from database.config import (
#     MYSQL_CONFIG,
#     RAZORPAY_KEY_ID,
#     RAZORPAY_KEY_SECRET
# )
# from helper.subscription_email_provider import send_subscription_email
# from helper.admin_email_provider import send_admin_payment_email


# # ===================== Razorpay =====================
# razorpay_client = razorpay.Client(
#     auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
# )

# CURRENCY_MULTIPLIER = {
#     "INR": 100,
#     "USD": 100
# }

# SUPPORTED_CURRENCIES = ["INR", "USD"]

# def create_razorpay_order(amount, currency="USD"):
#     if not amount or amount <= 0:
#         raise ValueError("Amount must be greater than 0")

#     currency = currency.upper()

#     if currency not in SUPPORTED_CURRENCIES:
#         raise ValueError(f"Unsupported currency: {currency}")

#     multiplier = CURRENCY_MULTIPLIER.get(currency, 100)
#     razorpay_amount = int(round(float(amount) * multiplier))

#     order_data = {
#         "amount": razorpay_amount,
#         "currency": currency,
#         "payment_capture": 1
#     }

#     return razorpay_client.order.create(order_data)

# def create_invoice_after_payment(transaction_id):
#     conn = get_connection()
#     cur = conn.cursor()

#     cur.execute("""
#         SELECT user_id, plan_name, amount, currency, billing_address
#         FROM subscriptions
#         WHERE transaction_id = %s
#     """, (transaction_id,))

#     row = cur.fetchone()
#     conn.close()

#     if not row:
#         return None

#     invoice_number = f"INV-{transaction_id}"
#     billing = json.loads(row["billing_address"] or "{}")

#     invoice_data = {
#     "invoice_number": invoice_number,
#     "date": datetime.now().strftime("%d %b %Y"),
#     "full_name": full_name,                     # 🔥 REQUIRED
#     "coupon_code": row.get("coupon_code"), 
#     "plan_name": plan_name,
#     "amount": float(amount),
#     "final_amount": float(amount),
#     "currency": currency,
#     "payment_method": "Razorpay",  # ✅ ADD THIS
#     "start_date": start_date.strftime("%d %b %Y") if start_date else "N/A",
#     "end_date": end_date.strftime("%d %b %Y") if end_date else "N/A",
#     "billing_lines": [
#         billing_address.get("address_line1", ""),
#         billing_address.get("address_line2", ""),
#         f"{billing_address.get('city', '')}, {billing_address.get('state', '')}",
#         billing_address.get("country", "")
#         ]
#     }

#     pdf_path = generate_invoice_pdf(invoice_data)
#     return pdf_path


# # ===================== DB =====================
# def get_connection():
#     return pymysql.connect(
#         host=MYSQL_CONFIG["host"],
#         port=MYSQL_CONFIG["port"],
#         user=MYSQL_CONFIG["user"],
#         password=MYSQL_CONFIG["password"],
#         database=MYSQL_CONFIG["database"],
#         cursorclass=pymysql.cursors.DictCursor
#     )


# # ===================== PLAN =====================
# def get_plan_details(plan_name):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT plan_code,
#                plan_name,
#                minutes_per_day,
#                validity_days,
#                is_trial,
#                plan_level
#         FROM subscription_plans
#         WHERE plan_name = %s
#           AND is_active = 1
#         LIMIT 1
#     """, (plan_name,))
#     plan = cur.fetchone()
#     conn.close()
#     return plan


# # ===================== SUB HELPERS =====================
# def get_active_subscription(user_id):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT plan_name, start_date, end_date
#         FROM subscriptions
#         WHERE user_id = %s
#           AND status = 'active'
#           AND NOW() BETWEEN start_date AND end_date
#         ORDER BY start_date DESC
#         LIMIT 1
#     """, (user_id,))
#     sub = cur.fetchone()
#     conn.close()
#     return sub


# def get_latest_subscription(user_id):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT plan_name, start_date, end_date, status
#         FROM subscriptions
#         WHERE user_id = %s
#         ORDER BY start_date DESC
#         LIMIT 1
#     """, (user_id,))
#     sub = cur.fetchone()
#     conn.close()
#     return sub


# def get_active_subscription_with_plan(user_id):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT s.subscription_id,
#                s.plan_name,
#                p.plan_level
#         FROM subscriptions s
#         JOIN subscription_plans p
#           ON s.plan_code = p.plan_code
#         WHERE s.user_id = %s
#           AND s.status = 'active'
#         ORDER BY s.start_date DESC
#         LIMIT 1
#     """, (user_id,))
#     row = cur.fetchone()
#     conn.close()
#     return row


# def get_today_voice_window(user_id, daily_limit_seconds):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT created_at
#         FROM conversation_history
#         WHERE user_id = %s
#           AND DATE(created_at) = CURDATE()
#           AND user_input NOT IN ('[SESSION STARTED]')
#         ORDER BY created_at ASC
#         LIMIT 1
#     """, (user_id,))
#     row = cur.fetchone()
#     conn.close()

#     if not row:
#         return None

#     start_time = row["created_at"]
#     end_time = start_time + timedelta(seconds=daily_limit_seconds)
#     remaining = int((end_time - datetime.now()).total_seconds())

#     return {
#         "start_time": start_time,
#         "end_time": end_time,
#         "remaining_seconds": max(0, remaining)
#     }


# # ======================= COUPON HELPERS ======================
# def get_coupon_details(code):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT *
#         FROM coupons
#         WHERE code = %s
#           AND is_active = 1
#           AND NOW() BETWEEN valid_from AND valid_to
#     """, (code,))
#     coupon = cur.fetchone()
#     conn.close()
#     return coupon


# def apply_coupon_on_amount(amount, coupon):
#     if amount < coupon["min_amount"]:
#         return None, None, f"Minimum amount should be {coupon['min_amount']}"

#     discount = 0

#     if coupon["discount_type"] == "PERCENT":
#         discount = (amount * coupon["discount_value"]) // 100
#         if coupon["max_discount"]:
#             discount = min(discount, coupon["max_discount"])

#     elif coupon["discount_type"] == "FLAT":
#         discount = coupon["discount_value"]

#     final_amount = max(0, amount - discount)
#     return final_amount, discount, None


# # ===================== START / UPGRADE =====================
# def start_subscription_controller():
#     data = request.json or {}

#     user_id = data.get("user_id")
#     plan_name = data.get("plan_name")
#     email = data.get("email")
#     full_name = data.get("full_name", "User")

#     billing = data.get("billing", {})
#     payment = data.get("payment", {})

#     payment_method = payment.get("method", "FREE")
#     raw_amount = payment.get("amount", 0)
#     currency = payment.get("currency", "USD").upper()

#     coupon_code = data.get("coupon_code")

#     try:
#         amount = float(raw_amount)
#     except (TypeError, ValueError):
#         amount = 0

#     if not user_id or not plan_name:
#         return jsonify({"error": "user_id and plan_name required"}), 400

#     if currency not in SUPPORTED_CURRENCIES:
#         return jsonify({"error": "Unsupported currency"}), 400

#     plan = get_plan_details(plan_name)
#     if not plan:
#         return jsonify({"error": "Invalid or inactive plan"}), 400

#     conn = get_connection()
#     cur = conn.cursor()

#     # ---------- Upgrade handling ----------
#     current = get_active_subscription_with_plan(user_id)
#     if current:
#         if plan["plan_level"] <= current["plan_level"]:
#             return jsonify({"error": "You can only upgrade to a higher plan"}), 400

#         cur.execute("""
#             UPDATE subscriptions
#             SET status = 'cancelled',
#                 end_date = NOW()
#             WHERE subscription_id = %s
#         """, (current["subscription_id"],))

#     # ---------- Apply Coupon ----------
#     discount_amount = 0
#     applied_coupon = None

#     if coupon_code:
#         coupon = get_coupon_details(coupon_code)
#         if not coupon:
#             return jsonify({"error": "Invalid or expired coupon"}), 400

#         final_amount, discount_amount, error = apply_coupon_on_amount(amount, coupon)
#         if error:
#             return jsonify({"error": error}), 400

#         amount = final_amount
#         applied_coupon = coupon_code

#         cur.execute("""
#             UPDATE coupons
#             SET used_count = used_count + 1
#             WHERE coupon_id = %s
#         """, (coupon["coupon_id"],))

#     # ---------- Razorpay Init ----------
#     gateway_order = None
#     status = "active"

#     if payment_method == "RAZORPAY" and amount > 0:
#         try:
#             gateway_order = create_razorpay_order(amount, currency)
#             status = "pending"
#         except Exception as e:
#             return jsonify({
#                 "error": "Failed to create Razorpay order",
#                 "details": str(e)
#             }), 500

#     # ---------- Create subscription ----------
#     start_date = datetime.now()
#     end_date = start_date + timedelta(days=plan["validity_days"])
#     transaction_id = f"TXN-{user_id}-{int(start_date.timestamp())}"

#     cur.execute("""
#         INSERT INTO subscriptions
#         (user_id,
#          plan_code,
#          plan_name,
#          amount,
#          currency,
#          status,
#          start_date,
#          end_date,
#          payment_method,
#          transaction_id,
#          billing_address,
#          coupon_code,
#          discount_amount,
#          gateway_payload)
#         VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
#     """, (
#         user_id,
#         plan["plan_code"],
#         plan_name,
#         amount,
#         currency,
#         status,
#         start_date,
#         end_date,
#         payment_method,
#         transaction_id,
#         json.dumps(billing),
#         applied_coupon,
#         discount_amount,
#         json.dumps(gateway_order)
#     ))

#     conn.commit()
#     conn.close()

#     # ---------- Email (only if FREE or already PAID) ----------
#     if email and status == "active":
#         send_subscription_email(
#             to_email=email,
#             full_name=full_name,
#             plan_name=plan_name,
#             amount=amount,
#             currency=currency,
#             start_date=start_date.strftime("%d %b %Y"),
#             end_date=end_date.strftime("%d %b %Y"),
#             payment_method=payment_method,
#             transaction_id=transaction_id
#         )

#     # ---------- UI RESPONSE ----------
#     return jsonify({
#         "status": status.upper(),
#         "transaction_id": transaction_id,
#         "plan": plan_name,
#         "original_amount": raw_amount,
#         "discount_amount": discount_amount,
#         "final_amount_paid": amount,
#         "coupon_applied": applied_coupon,
#         "payment_method": payment_method,
#         "currency": currency,
#         "razorpay_key": RAZORPAY_KEY_ID if gateway_order else None,
#         "gateway_order": gateway_order,
#         "valid_till": end_date.strftime("%d %b %Y")
#     }), 201


# # ===================== RAZORPAY VERIFY =====================


# def razorpay_verify_controller():
#     data = request.json or {}

#     print("🔥 Razorpay verify API HIT")
#     print("Payload:", data)

#     DEV_MODE = True  # 🔴 SET False IN PRODUCTION

#     if "transaction_id" not in data:
#         return jsonify({"success": False, "error": "transaction_id missing"}), 400

#     conn = get_connection()
#     cur = conn.cursor()

#     try:
#         # 1. GET SUBSCRIPTION DETAILS
#         cur.execute("""
#             SELECT user_id, plan_name, amount, currency, start_date, end_date, email_sent, billing_address
#             FROM subscriptions
#             WHERE transaction_id = %s
#         """, (data["transaction_id"],))
#         row = cur.fetchone()

#         if not row:
#             print("❌ No subscription found for this Transaction ID")
#             return jsonify({"success": False, "error": "Transaction not found"}), 404

#         print("✅ Subscription row found:", row)

#         user_id = row["user_id"]
#         plan_name = row["plan_name"]
#         amount = row["amount"]
#         currency = row["currency"]
#         start_date = row.get("start_date")
#         end_date = row.get("end_date")
#         email_sent = row["email_sent"]
#         billing_address = json.loads(row["billing_address"] or "{}")

#         # 2. UPDATE STATUS TO ACTIVE
#         if DEV_MODE:
#             cur.execute(
#                 "UPDATE subscriptions SET status = 'active' WHERE transaction_id = %s",
#                 (data["transaction_id"],)
#             )
#         else:
#             required = ["razorpay_order_id", "razorpay_payment_id", "razorpay_signature"]
#             if not all(k in data for k in required):
#                 return jsonify({"success": False, "error": "Missing Razorpay params"}), 400

#             try:
#                 razorpay_client.utility.verify_payment_signature({
#                     "razorpay_order_id": data["razorpay_order_id"],
#                     "razorpay_payment_id": data["razorpay_payment_id"],
#                     "razorpay_signature": data["razorpay_signature"]
#                 })
#                 cur.execute(
#                     "UPDATE subscriptions SET status = 'active' WHERE transaction_id = %s",
#                     (data["transaction_id"],)
#                 )
#             except Exception as e:
#                 print(f"❌ Signature verification failed: {e}")
#                 return jsonify({"success": False, "error": "Signature mismatch"}), 400

#         conn.commit()
#         print("✅ Subscription marked as ACTIVE")

#         # 3. CREATE INVOICE
#         invoice_number = f"INV-{data['transaction_id']}"

#         invoice_data = {
#             "invoice_number": invoice_number,
#             "date": datetime.now().strftime("%d %b %Y"),
#             "plan_name": plan_name,
#             "amount": float(amount),
#             "final_amount": float(amount),
#             "currency": currency,
#             "payment_method": "Razorpay",  # ✅ ADD THIS
#             "start_date": start_date.strftime("%d %b %Y") if start_date else "N/A",
#             "end_date": end_date.strftime("%d %b %Y") if end_date else "N/A",
#             "billing_lines": [
#                 billing_address.get("address_line1", ""),
#                 billing_address.get("address_line2", ""),
#                 f"{billing_address.get('city', '')}, {billing_address.get('state', '')}",
#                 billing_address.get("country", "")
#             ]
#         }

#         # 🔁 Prevent duplicate invoice insert
#         cur.execute(
#             "SELECT 1 FROM invoices WHERE transaction_id = %s",
#             (data["transaction_id"],)
#         )
#         exists = cur.fetchone()

#         if not exists:
#             cur.execute("""
#                 INSERT INTO invoices
#                 (user_id, transaction_id, invoice_number, plan_name, amount, currency, billing_address)
#                 VALUES (%s,%s,%s,%s,%s,%s,%s)
#             """, (
#                 user_id,
#                 data["transaction_id"],
#                 invoice_number,
#                 plan_name,
#                 amount,
#                 currency,
#                 json.dumps(billing_address)
#             ))
#             conn.commit()
#             print("🧾 Invoice DB record created")
#         else:
#             print("⚠️ Invoice already exists. Skipping insert.")

#         # Generate PDF
#         pdf_path = generate_invoice_pdf(invoice_data)
#         print("📄 Invoice PDF generated:", pdf_path)

#         invoice_url = pdf_path.replace("invoices/", "/static/invoices/")

#         # 4. HANDLE EMAIL SENDING
#         if email_sent == 1:
#             print("⚠️ Email already sent previously. Skipping.")
#             return jsonify({
#                 "success": True,
#                 "invoice_url": invoice_url
#             }), 200

#         # Fetch User Details
#         cur.execute("SELECT email, full_name FROM users WHERE user_id = %s", (user_id,))
#         user_row = cur.fetchone()

#         if not user_row:
#             print("❌ CRITICAL: User ID not found in users table.")
#             return jsonify({
#                 "success": True,
#                 "warning": "User not found, email skipped",
#                 "invoice_url": invoice_url
#             }), 200

#         user_email = user_row["email"]
#         full_name = user_row["full_name"]

#         try:
#             send_subscription_email(
#                 to_email=user_email,
#                 full_name=full_name,
#                 plan_name=plan_name,
#                 amount=amount,
#                 currency=currency,
#                 start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
#                 end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
#                 payment_method="Razorpay",
#                 transaction_id=data["transaction_id"],
#                 attachment_path=pdf_path
#             )

#             send_admin_payment_email(
#                 user_email=user_email,
#                 amount=amount,
#                 payment_id=data["transaction_id"],
#                 plan_name=plan_name
#             )

#             cur.execute(
#                 "UPDATE subscriptions SET email_sent = 1 WHERE transaction_id = %s",
#                 (data["transaction_id"],)
#             )
#             conn.commit()
#             print("📧 Emails sent and DB updated")

#         except Exception as e:
#             print(f"❌ EMAIL SENDING FAILED: {str(e)}")

#     except Exception as e:
#         print(f"❌ DATABASE/CONTROLLER ERROR: {str(e)}")
#         return jsonify({"success": False, "error": str(e)}), 500

#     finally:
#         if conn.open:
#             conn.close()

#     # 5. FINAL RESPONSE
#     return jsonify({
#         "success": True,
#         "invoice_url": invoice_url
#     }), 200


# # ===================== STATUS =====================
# def subscription_status_controller():
#     user_id = request.args.get("user_id")
#     if not user_id:
#         return jsonify({"error": "user_id required"}), 400

#     sub = get_latest_subscription(user_id)
#     if not sub:
#         return jsonify({"active": False, "voice_allowed": False}), 200

#     now = datetime.now()
#     if sub["status"] != "active" or not (sub["start_date"] <= now <= sub["end_date"]):
#         return jsonify({"active": False, "voice_allowed": False}), 200

#     plan = get_plan_details(sub["plan_name"])
#     if not plan:
#         return jsonify({"active": False, "voice_allowed": False}), 200

#     days_used = (now - sub["start_date"]).days
#     days_left = max(0, plan["validity_days"] - days_used)

#     if plan["minutes_per_day"] is None:
#         return jsonify({
#             "active": True,
#             "plan": plan["plan_name"],
#             "validity_days_left": days_left,
#             "daily_minutes_left": "unlimited",
#             "voice_allowed": True
#         }), 200

#     daily_limit_seconds = plan["minutes_per_day"] * 60
#     window = get_today_voice_window(user_id, daily_limit_seconds)

#     if not window:
#         remaining_seconds = daily_limit_seconds
#         voice_allowed = True
#     else:
#         remaining_seconds = window["remaining_seconds"]
#         voice_allowed = remaining_seconds > 0

#     return jsonify({
#         "active": True,
#         "plan": plan["plan_name"],
#         "validity_days_left": days_left,
#         "daily_minutes_left": remaining_seconds // 60,
#         "voice_allowed": voice_allowed
#     }), 200


# # ===================== COUPON VALIDATE =====================
# def validate_coupon_controller():
#     data = request.json or {}

#     coupon_code = data.get("coupon_code")
#     raw_amount = data.get("amount")

#     if not coupon_code:
#         return jsonify({"valid": False, "message": "coupon_code is required"}), 400

#     try:
#         amount = float(raw_amount)
#     except (TypeError, ValueError):
#         return jsonify({"valid": False, "message": "Invalid amount"}), 400

#     coupon = get_coupon_details(coupon_code)
#     if not coupon:
#         return jsonify({"valid": False, "message": "Invalid or expired coupon"}), 200

#     final_amount, discount_amount, error = apply_coupon_on_amount(amount, coupon)
#     if error:
#         return jsonify({"valid": False, "message": error}), 200

#     return jsonify({
#         "valid": True,
#         "coupon_code": coupon_code,
#         "discount_type": coupon["discount_type"],
#         "discount_value": coupon["discount_value"],
#         "discount_amount": discount_amount,
#         "original_amount": amount,
#         "final_amount": final_amount,
#         "message": "Coupon applied successfully"
#     }), 200


from flask import request, jsonify
from datetime import datetime, timedelta
from helper.invoice_generator import generate_invoice_pdf
import pymysql
import json
import razorpay


from database.config import (
    MYSQL_CONFIG,
    RAZORPAY_KEY_ID,
    RAZORPAY_KEY_SECRET
)
from helper.subscription_email_provider import send_subscription_email
from helper.admin_email_provider import send_admin_payment_email


# ===================== Razorpay =====================
razorpay_client = razorpay.Client(
    auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
)

CURRENCY_MULTIPLIER = {
    "INR": 100,
    "USD": 100
}

SUPPORTED_CURRENCIES = ["INR", "USD"]

def create_razorpay_order(amount, currency="USD"):
    if not amount or amount <= 0:
        raise ValueError("Amount must be greater than 0")

    currency = currency.upper()

    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {currency}")

    multiplier = CURRENCY_MULTIPLIER.get(currency, 100)
    razorpay_amount = int(round(float(amount) * multiplier))

    order_data = {
        "amount": razorpay_amount,
        "currency": currency,
        "payment_capture": 1
    }

    return razorpay_client.order.create(order_data)

def create_invoice_after_payment(transaction_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT user_id, plan_name, amount, currency, billing_address
        FROM subscriptions
        WHERE transaction_id = %s
    """, (transaction_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    invoice_number = f"INV-{transaction_id}"
    billing = json.loads(row["billing_address"] or "{}")

    # --- 🔥 CHANGE: Extract Mobile Number ---
    mobile_number = billing.get("mobile", "") 

    # Construct Billing Lines
    billing_lines_list = [
        billing.get("address_line1", ""),
        billing.get("address_line2", ""),
        f"{billing.get('city', '')}, {billing.get('state', '')}",
        billing.get("country", "")
    ]
    
    # Only add mobile line if it exists
    if mobile_number:
        billing_lines_list.append(f"Mobile: {mobile_number}")

    invoice_data = {
        "invoice_number": invoice_number,
        "date": datetime.now().strftime("%d %b %Y"),
        "full_name": "User", # Note: ideally fetch full_name from DB if needed here
        "coupon_code": row.get("coupon_code"), 
        "plan_name": row.get("plan_name"),
        "amount": float(row.get("amount")),
        "final_amount": float(row.get("amount")),
        "currency": row.get("currency"),
        "payment_method": "Razorpay",
        "start_date": "N/A", # Logic for dates can be refined if needed
        "end_date": "N/A",
        "billing_lines": billing_lines_list # ✅ Updated list with Mobile
    }

    pdf_path = generate_invoice_pdf(invoice_data)
    return pdf_path


# ===================== DB =====================
def get_connection():
    return pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        database=MYSQL_CONFIG["database"],
        cursorclass=pymysql.cursors.DictCursor
    )


# ===================== PLAN =====================
def get_plan_details(plan_name):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT plan_code,
               plan_name,
               minutes_per_day,
               validity_days,
               is_trial,
               plan_level
        FROM subscription_plans
        WHERE plan_name = %s
          AND is_active = 1
        LIMIT 1
    """, (plan_name,))
    plan = cur.fetchone()
    conn.close()
    return plan


# ===================== SUB HELPERS =====================
def get_active_subscription(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT plan_name, start_date, end_date
        FROM subscriptions
        WHERE user_id = %s
          AND status = 'active'
          AND NOW() BETWEEN start_date AND end_date
        ORDER BY start_date DESC
        LIMIT 1
    """, (user_id,))
    sub = cur.fetchone()
    conn.close()
    return sub


def get_latest_subscription(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT plan_name, start_date, end_date, status
        FROM subscriptions
        WHERE user_id = %s
        ORDER BY start_date DESC
        LIMIT 1
    """, (user_id,))
    sub = cur.fetchone()
    conn.close()
    return sub


def get_active_subscription_with_plan(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.subscription_id,
               s.plan_name,
               p.plan_level
        FROM subscriptions s
        JOIN subscription_plans p
          ON s.plan_code = p.plan_code
        WHERE s.user_id = %s
          AND s.status = 'active'
        ORDER BY s.start_date DESC
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def get_today_voice_window(user_id, daily_limit_seconds):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT created_at
        FROM conversation_history
        WHERE user_id = %s
          AND DATE(created_at) = CURDATE()
          AND user_input NOT IN ('[SESSION STARTED]')
        ORDER BY created_at ASC
        LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    start_time = row["created_at"]
    end_time = start_time + timedelta(seconds=daily_limit_seconds)
    remaining = int((end_time - datetime.now()).total_seconds())

    return {
        "start_time": start_time,
        "end_time": end_time,
        "remaining_seconds": max(0, remaining)
    }


# ======================= COUPON HELPERS ======================
def get_coupon_details(code):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM coupons
        WHERE code = %s
          AND is_active = 1
          AND NOW() BETWEEN valid_from AND valid_to
    """, (code,))
    coupon = cur.fetchone()
    conn.close()
    return coupon


def apply_coupon_on_amount(amount, coupon):
    if amount < coupon["min_amount"]:
        return None, None, f"Minimum amount should be {coupon['min_amount']}"

    discount = 0

    if coupon["discount_type"] == "PERCENT":
        discount = (amount * coupon["discount_value"]) // 100
        if coupon["max_discount"]:
            discount = min(discount, coupon["max_discount"])

    elif coupon["discount_type"] == "FLAT":
        discount = coupon["discount_value"]

    final_amount = max(0, amount - discount)
    return final_amount, discount, None


# ===================== START / UPGRADE =====================
def start_subscription_controller():
    data = request.json or {}

    user_id = data.get("user_id")
    plan_name = data.get("plan_name")
    email = data.get("email")
    full_name = data.get("full_name", "User")

    # 🔥 Ensure frontend sends 'mobile' inside 'billing' dictionary
    # Example: { "billing": { "address_line1": "...", "mobile": "9999999999" } }
    billing = data.get("billing", {})
    payment = data.get("payment", {})

    payment_method = payment.get("method", "FREE")
    raw_amount = payment.get("amount", 0)
    currency = payment.get("currency", "USD").upper()

    coupon_code = data.get("coupon_code")

    try:
        amount = float(raw_amount)
    except (TypeError, ValueError):
        amount = 0

    if not user_id or not plan_name:
        return jsonify({"error": "user_id and plan_name required"}), 400

    if currency not in SUPPORTED_CURRENCIES:
        return jsonify({"error": "Unsupported currency"}), 400

    plan = get_plan_details(plan_name)
    if not plan:
        return jsonify({"error": "Invalid or inactive plan"}), 400

    conn = get_connection()
    cur = conn.cursor()

    # ---------- Upgrade handling ----------
    current = get_active_subscription_with_plan(user_id)
    if current:
        if plan["plan_level"] <= current["plan_level"]:
            return jsonify({"error": "You can only upgrade to a higher plan"}), 400

        cur.execute("""
            UPDATE subscriptions
            SET status = 'cancelled',
                end_date = NOW()
            WHERE subscription_id = %s
        """, (current["subscription_id"],))

    # ---------- Apply Coupon ----------
    discount_amount = 0
    applied_coupon = None

    if coupon_code:
        coupon = get_coupon_details(coupon_code)
        if not coupon:
            return jsonify({"error": "Invalid or expired coupon"}), 400

        final_amount, discount_amount, error = apply_coupon_on_amount(amount, coupon)
        if error:
            return jsonify({"error": error}), 400

        amount = final_amount
        applied_coupon = coupon_code

        cur.execute("""
            UPDATE coupons
            SET used_count = used_count + 1
            WHERE coupon_id = %s
        """, (coupon["coupon_id"],))

    # ---------- Razorpay Init ----------
    gateway_order = None
    status = "active"

    if payment_method == "RAZORPAY" and amount > 0:
        try:
            gateway_order = create_razorpay_order(amount, currency)
            status = "pending"
        except Exception as e:
            return jsonify({
                "error": "Failed to create Razorpay order",
                "details": str(e)
            }), 500

    # ---------- Create subscription ----------
    start_date = datetime.now()
    end_date = start_date + timedelta(days=plan["validity_days"])
    transaction_id = f"TXN-{user_id}-{int(start_date.timestamp())}"

    cur.execute("""
        INSERT INTO subscriptions
        (user_id,
         plan_code,
         plan_name,
         amount,
         currency,
         status,
         start_date,
         end_date,
         payment_method,
         transaction_id,
         billing_address,
         coupon_code,
         discount_amount,
         gateway_payload)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        user_id,
        plan["plan_code"],
        plan_name,
        amount,
        currency,
        status,
        start_date,
        end_date,
        payment_method,
        transaction_id,
        json.dumps(billing), # This will save 'mobile' if frontend sent it
        applied_coupon,
        discount_amount,
        json.dumps(gateway_order)
    ))

    conn.commit()
    conn.close()

    # ---------- Email (only if FREE or already PAID) ----------
    if email and status == "active":
        send_subscription_email(
            to_email=email,
            full_name=full_name,
            plan_name=plan_name,
            amount=amount,
            currency=currency,
            start_date=start_date.strftime("%d %b %Y"),
            end_date=end_date.strftime("%d %b %Y"),
            payment_method=payment_method,
            transaction_id=transaction_id
        )

    # ---------- UI RESPONSE ----------
    return jsonify({
        "status": status.upper(),
        "transaction_id": transaction_id,
        "plan": plan_name,
        "original_amount": raw_amount,
        "discount_amount": discount_amount,
        "final_amount_paid": amount,
        "coupon_applied": applied_coupon,
        "payment_method": payment_method,
        "currency": currency,
        "razorpay_key": RAZORPAY_KEY_ID if gateway_order else None,
        "gateway_order": gateway_order,
        "valid_till": end_date.strftime("%d %b %Y"),
        "mobile": billing.get("mobile")
    }), 201


# ===================== RAZORPAY VERIFY =====================


def razorpay_verify_controller():
    data = request.json or {}

    print("🔥 Razorpay verify API HIT")
    print("Payload:", data)

    DEV_MODE = True  # 🔴 SET False IN PRODUCTION

    if "transaction_id" not in data:
        return jsonify({"success": False, "error": "transaction_id missing"}), 400

    conn = get_connection()
    cur = conn.cursor()

    try:
        # 1. GET SUBSCRIPTION DETAILS
        cur.execute("""
            SELECT user_id, plan_name, amount, currency, start_date, end_date, email_sent, billing_address
            FROM subscriptions
            WHERE transaction_id = %s
        """, (data["transaction_id"],))
        row = cur.fetchone()

        if not row:
            print("❌ No subscription found for this Transaction ID")
            return jsonify({"success": False, "error": "Transaction not found"}), 404

        print("✅ Subscription row found:", row)

        user_id = row["user_id"]
        plan_name = row["plan_name"]
        amount = row["amount"]
        currency = row["currency"]
        start_date = row.get("start_date")
        end_date = row.get("end_date")
        email_sent = row["email_sent"]
        billing_address = json.loads(row["billing_address"] or "{}")

        # 2. UPDATE STATUS TO ACTIVE
        if DEV_MODE:
            cur.execute(
                "UPDATE subscriptions SET status = 'active' WHERE transaction_id = %s",
                (data["transaction_id"],)
            )
        else:
            required = ["razorpay_order_id", "razorpay_payment_id", "razorpay_signature"]
            if not all(k in data for k in required):
                return jsonify({"success": False, "error": "Missing Razorpay params"}), 400

            try:
                razorpay_client.utility.verify_payment_signature({
                    "razorpay_order_id": data["razorpay_order_id"],
                    "razorpay_payment_id": data["razorpay_payment_id"],
                    "razorpay_signature": data["razorpay_signature"]
                })
                cur.execute(
                    "UPDATE subscriptions SET status = 'active' WHERE transaction_id = %s",
                    (data["transaction_id"],)
                )
            except Exception as e:
                print(f"❌ Signature verification failed: {e}")
                return jsonify({"success": False, "error": "Signature mismatch"}), 400

        conn.commit()
        print("✅ Subscription marked as ACTIVE")

        # 3. CREATE INVOICE
        invoice_number = f"INV-{data['transaction_id']}"
        
        # --- 🔥 CHANGE: Extract Mobile Number for Invoice ---
        mobile_number = billing_address.get("mobile", "")

        billing_lines_list = [
            billing_address.get("address_line1", ""),
            billing_address.get("address_line2", ""),
            f"{billing_address.get('city', '')}, {billing_address.get('state', '')}",
            billing_address.get("country", "")
        ]

        if mobile_number:
            billing_lines_list.append(f"Mobile: {mobile_number}")

        invoice_data = {
            "invoice_number": invoice_number,
            "date": datetime.now().strftime("%d %b %Y"),
            "plan_name": plan_name,
            "amount": float(amount),
            "final_amount": float(amount),
            "currency": currency,
            "payment_method": "Razorpay",
            "start_date": start_date.strftime("%d %b %Y") if start_date else "N/A",
            "end_date": end_date.strftime("%d %b %Y") if end_date else "N/A",
            "billing_lines": billing_lines_list # ✅ Added mobile to lines
        }

        # 🔁 Prevent duplicate invoice insert
        cur.execute(
            "SELECT 1 FROM invoices WHERE transaction_id = %s",
            (data["transaction_id"],)
        )
        exists = cur.fetchone()

        if not exists:
            cur.execute("""
                INSERT INTO invoices
                (user_id, transaction_id, invoice_number, plan_name, amount, currency, billing_address)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                user_id,
                data["transaction_id"],
                invoice_number,
                plan_name,
                amount,
                currency,
                json.dumps(billing_address)
            ))
            conn.commit()
            print("🧾 Invoice DB record created")
        else:
            print("⚠️ Invoice already exists. Skipping insert.")

        # Generate PDF
        pdf_path = generate_invoice_pdf(invoice_data)
        print("📄 Invoice PDF generated:", pdf_path)

        invoice_url = pdf_path.replace("invoices/", "/static/invoices/")

        # 4. HANDLE EMAIL SENDING
        if email_sent == 1:
            print("⚠️ Email already sent previously. Skipping.")
            return jsonify({
                "success": True,
                "invoice_url": invoice_url
            }), 200

        # Fetch User Details
        cur.execute("SELECT email, full_name FROM users WHERE user_id = %s", (user_id,))
        user_row = cur.fetchone()

        if not user_row:
            print("❌ CRITICAL: User ID not found in users table.")
            return jsonify({
                "success": True,
                "warning": "User not found, email skipped",
                "invoice_url": invoice_url
            }), 200

        user_email = user_row["email"]
        full_name = user_row["full_name"]

        try:
            send_subscription_email(
                to_email=user_email,
                full_name=full_name,
                plan_name=plan_name,
                amount=amount,
                currency=currency,
                start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
                payment_method="Razorpay",
                transaction_id=data["transaction_id"],
                attachment_path=pdf_path
            )

            send_admin_payment_email(
                user_email=user_email,
                amount=amount,
                payment_id=data["transaction_id"],
                plan_name=plan_name
            )

            cur.execute(
                "UPDATE subscriptions SET email_sent = 1 WHERE transaction_id = %s",
                (data["transaction_id"],)
            )
            conn.commit()
            print("📧 Emails sent and DB updated")

        except Exception as e:
            print(f"❌ EMAIL SENDING FAILED: {str(e)}")

    except Exception as e:
        print(f"❌ DATABASE/CONTROLLER ERROR: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if conn.open:
            conn.close()

    # 5. FINAL RESPONSE
    return jsonify({
        "success": True,
        "invoice_url": invoice_url
    }), 200


# ===================== STATUS =====================
def subscription_status_controller():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    sub = get_latest_subscription(user_id)
    if not sub:
        return jsonify({"active": False, "voice_allowed": False}), 200

    now = datetime.now()
    if sub["status"] != "active" or not (sub["start_date"] <= now <= sub["end_date"]):
        return jsonify({"active": False, "voice_allowed": False}), 200

    plan = get_plan_details(sub["plan_name"])
    if not plan:
        return jsonify({"active": False, "voice_allowed": False}), 200

    days_used = (now - sub["start_date"]).days
    days_left = max(0, plan["validity_days"] - days_used)

    if plan["minutes_per_day"] is None:
        return jsonify({
            "active": True,
            "plan": plan["plan_name"],
            "validity_days_left": days_left,
            "daily_minutes_left": "unlimited",
            "voice_allowed": True
        }), 200

    daily_limit_seconds = plan["minutes_per_day"] * 60
    window = get_today_voice_window(user_id, daily_limit_seconds)

    if not window:
        remaining_seconds = daily_limit_seconds
        voice_allowed = True
    else:
        remaining_seconds = window["remaining_seconds"]
        voice_allowed = remaining_seconds > 0

    return jsonify({
        "active": True,
        "plan": plan["plan_name"],
        "validity_days_left": days_left,
        "daily_minutes_left": remaining_seconds // 60,
        "voice_allowed": voice_allowed
    }), 200


# ===================== COUPON VALIDATE =====================
def validate_coupon_controller():
    data = request.json or {}

    coupon_code = data.get("coupon_code")
    raw_amount = data.get("amount")

    if not coupon_code:
        return jsonify({"valid": False, "message": "coupon_code is required"}), 400

    try:
        amount = float(raw_amount)
    except (TypeError, ValueError):
        return jsonify({"valid": False, "message": "Invalid amount"}), 400

    coupon = get_coupon_details(coupon_code)
    if not coupon:
        return jsonify({"valid": False, "message": "Invalid or expired coupon"}), 200

    final_amount, discount_amount, error = apply_coupon_on_amount(amount, coupon)
    if error:
        return jsonify({"valid": False, "message": error}), 200

    return jsonify({
        "valid": True,
        "coupon_code": coupon_code,
        "discount_type": coupon["discount_type"],
        "discount_value": coupon["discount_value"],
        "discount_amount": discount_amount,
        "original_amount": amount,
        "final_amount": final_amount,
        "message": "Coupon applied successfully"
    }), 200