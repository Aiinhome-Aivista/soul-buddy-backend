# from flask import request, jsonify
# from datetime import datetime, timedelta
# import pymysql
# import json  # ✅ IMPORTANT
# from database.config import MYSQL_CONFIG
# from helper.subscription_email_provider import send_subscription_email


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
#         SELECT plan_name,
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
# def get_active_subscription_with_plan(user_id):
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT s.subscription_id,
#                s.plan_name,
#                p.plan_level
#         FROM subscriptions s
#         JOIN subscription_plans p
#           ON s.plan_name = p.plan_name
#         WHERE s.user_id = %s
#           AND s.status = 'active'
#         ORDER BY s.start_date DESC
#         LIMIT 1
#     """, (user_id,))
#     row = cur.fetchone()
#     conn.close()
#     return row


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
#     amount = payment.get("amount", 0)
#     currency = payment.get("currency", "USD")

#     if not user_id or not plan_name:
#         return jsonify({"error": "user_id and plan_name required"}), 400

#     plan = get_plan_details(plan_name)
#     if not plan:
#         return jsonify({"error": "Invalid or inactive plan"}), 400

#     conn = get_connection()
#     cur = conn.cursor()

#     # ---------- Upgrade handling ----------
#     current = get_active_subscription_with_plan(user_id)
#     if current:
#         if plan["plan_level"] <= current["plan_level"]:
#             return jsonify({
#                 "error": "You can only upgrade to a higher plan"
#             }), 400

#         # Expire previous subscription
#         cur.execute("""
#             UPDATE subscriptions
#             SET status = 'cancelled',
#                 end_date = NOW()
#             WHERE subscription_id = %s
#         """, (current["subscription_id"],))

#     # ---------- Create new subscription ----------
#     start_date = datetime.now()
#     end_date = start_date + timedelta(days=plan["validity_days"])
#     transaction_id = f"TXN-{user_id}-{int(start_date.timestamp())}"

#     cur.execute("""
#         INSERT INTO subscriptions
#         (user_id,
#          plan_name,
#          amount,
#          currency,
#          status,
#          start_date,
#          end_date,
#          payment_method,
#          transaction_id,
#          billing_address)
#         VALUES (%s,%s,%s,%s,'active',%s,%s,%s,%s,%s)
#     """, (
#         user_id,
#         plan_name,
#         amount,
#         currency,
#         start_date,
#         end_date,
#         payment_method,
#         transaction_id,
#         json.dumps(billing)   # ✅ FIXED (VALID JSON)
#     ))

#     conn.commit()
#     conn.close()

#     # ---------- Email ----------
#     if email:
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

#     # ---------- UI RESPONSE (Payment Success Card) ----------
#     return jsonify({
#         "status": "PAID" if amount > 0 else "FREE",
#         "transaction_id": transaction_id,
#         "date": start_date.strftime("%d/%m/%Y, %I:%M %p"),
#         "plan": plan_name,
#         "payment_method": payment_method,
#         "amount_paid": amount,
#         "currency": currency,
#         "billing": billing,
#         "valid_till": end_date.strftime("%d %b %Y")
#     }), 201


# # ===================== STATUS =====================
# def subscription_status_controller():
#     user_id = request.args.get("user_id")
#     if not user_id:
#         return jsonify({"error": "user_id required"}), 400

#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute("""
#         SELECT plan_name, end_date
#         FROM subscriptions
#         WHERE user_id = %s
#           AND status = 'active'
#         ORDER BY start_date DESC
#         LIMIT 1
#     """, (user_id,))
#     sub = cur.fetchone()
#     conn.close()

#     if not sub:
#         return jsonify({"active": False}), 200

#     return jsonify({
#         "active": True,
#         "plan": sub["plan_name"],
#         "validity_days_left": max(
#             0,
#             (sub["end_date"] - datetime.now()).days
#         )
#     }), 200


from flask import request, jsonify
from datetime import datetime, timedelta
import pymysql
import json
from database.config import MYSQL_CONFIG
from helper.subscription_email_provider import send_subscription_email


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
    """
    Voice window starts from FIRST conversation today.
    """
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
    """
    Returns: (final_amount, discount_amount, error_message)
    """
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

    billing = data.get("billing", {})
    payment = data.get("payment", {})

    payment_method = payment.get("method", "FREE")
    raw_amount = payment.get("amount", 0)
    currency = payment.get("currency", "USD")

    coupon_code = data.get("coupon_code")

    # ---- Amount safety ----
    try:
        amount = int(raw_amount)
    except (TypeError, ValueError):
        amount = 0

    if not user_id or not plan_name:
        return jsonify({"error": "user_id and plan_name required"}), 400

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
         discount_amount)
        VALUES (%s,%s,%s,%s,%s,'active',%s,%s,%s,%s,%s,%s,%s)
    """, (
        user_id,
        plan["plan_code"],      # 🔴 REQUIRED FIX
        plan_name,
        amount,
        currency,
        start_date,
        end_date,
        payment_method,
        transaction_id,
        json.dumps(billing),
        applied_coupon,
        discount_amount
    ))

    conn.commit()
    conn.close()

    # ---------- Email ----------
    if email:
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
        "status": "PAID" if amount > 0 else "FREE",
        "transaction_id": transaction_id,
        "plan": plan_name,
        "original_amount": int(raw_amount) if str(raw_amount).isdigit() else raw_amount,
        "discount_amount": discount_amount,
        "final_amount_paid": amount,
        "coupon_applied": applied_coupon,
        "payment_method": payment_method,
        "currency": currency,
        "valid_till": end_date.strftime("%d %b %Y")
    }), 201


# ===================== STATUS =====================
def subscription_status_controller():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    sub = get_latest_subscription(user_id)
    if not sub:
        return jsonify({
            "active": False,
            "voice_allowed": False
        }), 200

    # Check if subscription is currently valid
    now = datetime.now()
    if sub["status"] != "active" or not (sub["start_date"] <= now <= sub["end_date"]):
        return jsonify({
            "active": False,
            "voice_allowed": False
        }), 200

    plan = get_plan_details(sub["plan_name"])
    if not plan:
        return jsonify({
            "active": False,
            "voice_allowed": False
        }), 200

    # ---- Validity ----
    days_used = (now - sub["start_date"]).days
    days_left = max(0, plan["validity_days"] - days_used)

    # ---- Daily Voice Minutes ----
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


def validate_coupon_controller():
    data = request.json or {}

    coupon_code = data.get("coupon_code")
    raw_amount = data.get("amount")

    # ---- Basic checks ----
    if not coupon_code:
        return jsonify({
            "valid": False,
            "message": "coupon_code is required"
        }), 400

    try:
        amount = int(raw_amount)
    except (TypeError, ValueError):
        return jsonify({
            "valid": False,
            "message": "Invalid amount"
        }), 400

    # ---- Fetch coupon ----
    coupon = get_coupon_details(coupon_code)
    if not coupon:
        return jsonify({
            "valid": False,
            "message": "Invalid or expired coupon"
        }), 200

    # ---- Apply coupon logic ----
    final_amount, discount_amount, error = apply_coupon_on_amount(amount, coupon)

    if error:
        return jsonify({
            "valid": False,
            "message": error
        }), 200

    # ---- Success response ----
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
