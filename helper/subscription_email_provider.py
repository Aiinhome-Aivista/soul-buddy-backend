import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database.config import NOREPLY_EMAIL, NOREPLY_PASSWORD

def send_subscription_email(
    to_email,
    full_name,
    plan_name,
    amount,
    currency,
    start_date,
    end_date,
    payment_method,
    transaction_id,
    attachment_path=None 
):
    try:
        # 1. Setup Email Headers
        msg = MIMEMultipart("alternative")
        msg["From"] = NOREPLY_EMAIL
        msg["To"] = to_email
        msg["Subject"] = "Confirmation: Your SoulJunction Subscription is Active"

        # 2. Professional Plain Text (Fallback)
        text_body = f"""
        Dear {full_name},

        We are pleased to confirm that your subscription to the {plan_name} plan has been successfully activated.

        Here are your subscription details:
        --------------------------------------------------
        Plan:           {plan_name}
        Amount Paid:    {currency} {amount}
        Payment Method: {payment_method}
        Transaction ID: {transaction_id}
        Start Date:     {start_date}
        End Date:       {end_date}
        --------------------------------------------------

        You can now access all premium features in your dashboard.

        Thank you for being a part of SoulJunction.

        Sincerely,
        The SoulJunction Team
        https://souljunction.life
        """

        # 3. Premium HTML Design
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f8f9fa; margin: 0; padding: 0; }}
            .container {{ max-width: 600px; margin: 40px auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); overflow: hidden; }}
            .header {{ background-color: #2c3e50; padding: 30px; text-align: center; }}
            .header h1 {{ color: #ffffff; margin: 0; font-size: 24px; font-weight: 300; letter-spacing: 1px; }}
            .content {{ padding: 40px 30px; color: #555555; line-height: 1.6; }}
            .receipt-box {{ background-color: #f1f4f6; border-radius: 6px; padding: 20px; margin: 25px 0; border: 1px solid #e1e4e8; }}
            .row {{ display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 14px; }}
            .row:last-child {{ margin-bottom: 0; }}
            .label {{ color: #7f8c8d; font-weight: 500; }}
            .value {{ color: #2c3e50; font-weight: 600; text-align: right; }}
            .button {{ display: block; width: 200px; margin: 30px auto; padding: 12px; background-color: #27ae60; color: #ffffff; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold; }}
            .footer {{ background-color: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #95a5a6; border-top: 1px solid #eeeeee; }}
            .status-active {{ color: #27ae60; font-weight: bold; }}
          </style>
        </head>
        <body>
          <div class="container">
            
            <div class="header">
              <h1>SOULJUNCTION</h1>
            </div>

            <div class="content">
              <p>Dear <strong>{full_name}</strong>,</p>
              <p>We are thrilled to welcome you. Your subscription has been successfully processed, and your account is now fully active.</p>

              <div class="receipt-box">
                <div class="row">
                  <span class="label">Plan Name: </span>
                  <span class="value">  {plan_name}</span>
                </div>
                <div class="row">
                  <span class="label">Status: </span>
                  <span class="value status-active">Active</span>
                </div>
                <hr style="border: 0; border-top: 1px solid #dcdde1; margin: 10px 0;">
                <div class="row">
                  <span class="label">Amount: </span>
                  <span class="value">  {currency} {amount}</span>
                </div>
                <div class="row">
                  <span class="label">Date: </span>
                  <span class="value">  {start_date}</span>
                </div>
                <div class="row">
                  <span class="label">Transaction ID: </span>
                  <span class="value" style="font-family: monospace;">  {transaction_id}</span>
                </div>
              </div>

              <p>Your subscription is valid until <strong>{end_date}</strong>. You can view your invoice history and manage your settings at any time from your profile.</p>

              <a href="https://souljunction.life/" class="button">Go to Dashboard</a>
              
              <p style="margin-top: 30px;">Warm regards,<br>The SoulJunction Team</p>
            </div>

            <div class="footer">
              <p>&copy; 2026 Soul Junction. All rights reserved.</p>
              <p>Need help? Contact <a href="mailto:support@souljunction.life" style="color: #2c3e50;">support@souljunction.life</a></p>
            </div>
          </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # 4. Send using Namecheap (SSL Port 465)
        with smtplib.SMTP_SSL("mail.souljunction.life", 465, timeout=20) as server:
            server.login(NOREPLY_EMAIL, NOREPLY_PASSWORD)
            server.sendmail(
                NOREPLY_EMAIL,
                [to_email],  # ✅ explicit recipient
                msg.as_string()
            )

        print("✅ User email sent to:", to_email)
        return True

    except Exception as e:
        print(f"Subscription Email Error: {e}")
        return False