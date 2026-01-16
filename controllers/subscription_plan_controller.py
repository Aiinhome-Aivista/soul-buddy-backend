# from flask import request, jsonify
# from datetime import datetime, timedelta
# import pymysql
# from database.config import MYSQL_CONFIG


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


# # ===================== PLAN HELPERS =====================
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

#     row = cur.fetchone()
#     conn.close()
#     return row


# # ===================== SUBSCRIPTION HELPERS =====================
# def get_active_subscription(user_id):
#     conn = get_connection()
#     cur = conn.cursor()

#     cur.execute("""
#         SELECT plan_name, start_date, end_date
#         FROM subscriptions
#         WHERE user_id = %s
#           AND status = 'active'
#           AND NOW() BETWEEN start_date AND end_date
#         LIMIT 1
#     """, (user_id,))

#     row = cur.fetchone()
#     conn.close()
#     return row

# def get_active_subscription_with_plan(user_id):
#     conn = get_connection()
#     cur = conn.cursor()

#     cur.execute("""
#         SELECT s.subscription_id,
#                s.plan_name,
#                s.start_date,
#                s.end_date,
#                p.plan_level
#         FROM subscriptions s
#         JOIN subscription_plans p
#           ON s.plan_name = p.plan_name
#         WHERE s.user_id = %s
#           AND s.status = 'active'
#           AND NOW() BETWEEN s.start_date AND s.end_date
#         LIMIT 1
#     """, (user_id,))

#     row = cur.fetchone()
#     conn.close()
#     return row


# def get_today_voice_window(user_id, daily_limit_seconds):
#     """
#     Voice window starts from FIRST conversation today.
#     """
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


# # ===================== CONTROLLERS =====================
# def start_subscription_controller():
#     data = request.json or {}
#     user_id = data.get("user_id")
#     plan_name = data.get("plan_name")

#     if not user_id or not plan_name:
#         return jsonify({"error": "user_id and plan_name are required"}), 400

#     plan = get_plan_details(plan_name)
#     if not plan:
#         return jsonify({"error": "Invalid or inactive plan"}), 400

#     conn = get_connection()
#     cur = conn.cursor()

#     # Check existing active subscription
#     cur.execute("""
#         SELECT subscription_id
#         FROM subscriptions
#         WHERE user_id = %s
#           AND status = 'active'
#           AND NOW() BETWEEN start_date AND end_date
#         LIMIT 1
#     """, (user_id,))

#     # if cur.fetchone():
#     #     conn.close()
#     #     return jsonify({"message": "Subscription already active"}), 200

#     current_sub = get_active_subscription_with_plan(user_id)

#     if current_sub:
#         # Fetch new plan level
#         if plan["plan_level"] <= current_sub["plan_level"]:
#             return jsonify({
#                 "error": "You can only upgrade to a higher plan"
#         }), 400

#     # ✅ Expire old subscription (UPGRADE)
#         cur.execute("""
#         UPDATE subscriptions
#         SET status = 'inactive',
#             end_date = NOW()
#         WHERE subscription_id = %s
#         """, (current_sub["subscription_id"],))



#     start_date = datetime.now()
#     end_date = start_date + timedelta(days=plan["validity_days"])

#     cur.execute("""
#         INSERT INTO subscriptions
#         (user_id, plan_name, amount, currency, status, start_date, end_date)
#         VALUES (%s, %s, %s, %s, 'active', %s, %s)
#     """, (
#         user_id,
#         plan["plan_name"],
#         0,
#         "INR",
#         start_date,
#         end_date
#     ))

#     conn.commit()
#     conn.close()

#     return jsonify({
#         "message": "Subscription activated",
#         "plan": plan["plan_name"],
#         "is_trial": bool(plan["is_trial"]),
#         "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
#         "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
#     }), 201


# def subscription_status_controller():
#     user_id = request.args.get("user_id")

#     if not user_id:
#         return jsonify({"error": "user_id is required"}), 400

#     sub = get_active_subscription(user_id)
#     if not sub:
#         return jsonify({
#             "active": False,
#             "voice_allowed": False
#         }), 200

#     plan = get_plan_details(sub["plan_name"])
#     if not plan:
#         return jsonify({
#             "active": False,
#             "voice_allowed": False
#         }), 200

#     # ---- Trial / Validity ----
#     days_used = (datetime.now() - sub["start_date"]).days
#     days_left = max(0, plan["validity_days"] - days_used)

#     # ---- Daily Voice Minutes ----
#     if plan["minutes_per_day"] is None:
#         # Unlimited minutes
#         return jsonify({
#             "active": True,
#             "plan": plan["plan_name"],
#             "trial_days_left": days_left,
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
#         "trial_days_left": days_left,
#         "daily_minutes_left": remaining_seconds // 60,
#         "voice_allowed": voice_allowed
#     }), 200


from flask import request, jsonify
from datetime import datetime, timedelta
import pymysql
from database.config import MYSQL_CONFIG
from helper.subscription_email_provider import send_subscription_email


# ================= DB =================
def get_connection():
    return pymysql.connect(
        **MYSQL_CONFIG,
        cursorclass=pymysql.cursors.DictCursor
    )


# ================= PLAN =================
def get_plan_by_code(plan_code):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT *
        FROM subscription_plans
        WHERE plan_code=%s AND is_active=1
    """, (plan_code,))
    plan = cur.fetchone()
    conn.close()
    return plan


# ================= ACTIVE SUB =================
def get_active_subscription(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, p.plan_level
        FROM subscriptions s
        JOIN subscription_plans p ON s.plan_code = p.plan_code
        WHERE s.user_id=%s AND s.status='active'
        ORDER BY s.start_date DESC
        LIMIT 1
    """, (user_id,))
    sub = cur.fetchone()
    conn.close()
    return sub


# ================= CHECKOUT =================
def initiate_checkout():
    data = request.json
    plan_code = data.get("plan_code")

    plan = get_plan_by_code(plan_code)
    if not plan:
        return jsonify({"error": "Invalid plan"}), 400

    return jsonify({
        "plan_name": plan["plan_name"],
        "original_price": float(plan["original_price"]),
        "discount_percent": plan["discount_percent"],
        "final_amount": float(plan["offer_price"]),
        "currency": "INR",
        "validity_days": plan["validity_days"]
    })


# ================= PAYMENT CONFIRM (MOCK) =================
def confirm_payment():
    data = request.json

    user_id = data["user_id"]
    email = data["email"]
    full_name = data["full_name"]
    plan_code = data["plan_code"]
    payment_method = data.get("payment_method", "UPI")

    plan = get_plan_by_code(plan_code)
    if not plan:
        return jsonify({"error": "Invalid plan"}), 400

    current = get_active_subscription(user_id)

    conn = get_connection()
    cur = conn.cursor()

    # ----- Upgrade logic -----
    if current:
        if plan["plan_level"] <= current["plan_level"]:
            return jsonify({"error": "Only upgrade allowed"}), 400

        cur.execute("""
            UPDATE subscriptions
            SET status='cancelled', end_date=NOW()
            WHERE subscription_id=%s
        """, (current["subscription_id"],))

    start_date = datetime.now()
    end_date = start_date + timedelta(days=plan["validity_days"])
    transaction_id = f"TXN-{int(datetime.now().timestamp())}"

    cur.execute("""
        INSERT INTO subscriptions (
            user_id,
            plan_code,
            plan_name,
            amount,
            currency,
            status,
            start_date,
            end_date,
            payment_method,
            transaction_id
        )
        VALUES (%s,%s,%s,%s,%s,'active',%s,%s,%s,%s)
    """, (
        user_id,
        plan["plan_code"],
        plan["plan_name"],
        plan["offer_price"],
        "INR",
        start_date,
        end_date,
        payment_method,
        transaction_id
    ))

    conn.commit()
    conn.close()

    # ----- EMAIL -----
    send_subscription_email(
        to_email=email,
        full_name=full_name,
        plan_name=plan["plan_name"],
        amount=plan["offer_price"],
        currency="INR",
        start_date=start_date.strftime("%d %b %Y"),
        end_date=end_date.strftime("%d %b %Y"),
        payment_method=payment_method,
        transaction_id=transaction_id
    )

    return jsonify({
        "status": "success",
        "transaction_id": transaction_id,
        "plan": plan["plan_name"],
        "payment_method": payment_method,
        "amount_paid": plan["offer_price"],
        "currency": "INR",
        "date": start_date.strftime("%d %b %Y, %I:%M %p"),
        "valid_till": end_date.strftime("%d %b %Y")
    }), 200
