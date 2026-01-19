# import smtplib
# import random
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from flask import request, jsonify

# # Replace these with your actual config imports

# from database.config import GMAIL_USER, GMAIL_APP_PASSWORD


# # Temporary storage (Use Redis/DB in production)
# otp_storage = {}

# def send_email_otp(user_email, otp_code):
#     """Sends a professional HTML email with the OTP."""
    
#     # 1. Setup the Message Container (MIMEMultipart)
#     msg = MIMEMultipart("alternative")
#     msg['Subject'] = "Your SoulJunction Verification Code"
#     msg['From'] = GMAIL_USER
#     msg['To'] = user_email

#     # 2. Define the HTML Design
#     # You can change colors (hex codes) to match your brand exactly.
#     html_content = f"""
#     <html>
#       <head></head>
#       <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
#         <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
          
#           <h2 style="color: #333333; text-align: center;">SoulJunction Verification</h2>
#           <hr style="border: 0; border-top: 1px solid #eeeeee;">
          
#           <p style="font-size: 16px; color: #555555;">Hello,</p>
#           <p style="font-size: 16px; color: #555555;">Use the code below to complete your sign-in process. This code is valid for 5 minutes.</p>
          
#           <div style="text-align: center; margin: 30px 0;">
#             <span style="font-size: 32px; font-weight: bold; color: #2c3e50; letter-spacing: 5px; background-color: #e8f0fe; padding: 10px 20px; border-radius: 5px; border: 1px dashed #2c3e50;">
#               {otp_code}
#             </span>
#           </div>

#           <p style="font-size: 14px; color: #888888; text-align: center;">
#             If you didn't request this, you can safely ignore this email.
#           </p>
          
#           <hr style="border: 0; border-top: 1px solid #eeeeee; margin-top: 30px;">
#           <div style="text-align: center; margin-top: 20px;">
#             <img src="logo/sblogo3.png" alt="Soul Junction Logo" style="width: 80px; height: auto; opacity: 0.8;">
#             <p style="font-size: 12px; color: #aaaaaa; margin-top: 5px;">&copy; 2024 Soul Junction Inc. All rights reserved.</p>
#           </div>
          
#         </div>
#       </body>
#     </html>
#     """

#     # 3. Attach the HTML to the email
#     part_html = MIMEText(html_content, "html")
#     msg.attach(part_html)

#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
#             server.send_message(msg)
#         return True
#     except Exception as e:
#         print(f"Error sending email: {e}")
#         return False
# # --- Controller 1: Send OTP ---
# def send_otp_controller():
#     data = request.get_json()
#     email = data.get('email')

#     if not email:
#         return jsonify({"error": "Email is required"}), 400

#     # Generate 6-digit OTP
#     otp = str(random.randint(100000, 999999))
    
#     # Store OTP
#     otp_storage[email] = otp

#     # Send Email
#     if send_email_otp(email, otp):
#         return jsonify({"success": True, "message": f"OTP sent to {email}"}), 200
#     else:
#         return jsonify({"success": False, "message": "Failed to send email"}), 500

# # --- Controller 2: Verify OTP ---
# def verify_otp_controller():
#     data = request.get_json()
#     email = data.get('email')
#     user_otp = data.get('otp')

#     if not email or not user_otp:
#         return jsonify({"error": "Email and OTP are required"}), 400

#     stored_otp = otp_storage.get(email)

#     if not stored_otp:
#         return jsonify({"error": "No OTP found or expired"}), 400

#     if stored_otp == user_otp:
#         del otp_storage[email]  # Clear OTP after success
#         return jsonify({"success": True, "message": "Verification Successful!"}), 200
#     else:
#         return jsonify({"error": "Invalid OTP"}), 400


import smtplib
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import request, jsonify

# Update imports to match your config file variables for Namecheap
from database.config import NOREPLY_EMAIL, NOREPLY_PASSWORD

# Temporary storage (Use Redis/DB in production)
otp_storage = {}

def send_email_otp(user_email, otp_code):
    """Sends a professional HTML email with the OTP using Namecheap SMTP."""
    
    # 1. Setup the Message Container
    msg = MIMEMultipart("alternative")
    msg['Subject'] = "Your SoulJunction Verification Code"
    msg['From'] = NOREPLY_EMAIL
    msg['To'] = user_email

    # 2. Define the HTML Design
    html_content = f"""
    <html>
      <head></head>
      <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
          
          <h2 style="color: #333333; text-align: center;">SoulJunction Verification</h2>
          <hr style="border: 0; border-top: 1px solid #eeeeee;">
          
          <p style="font-size: 16px; color: #555555;">Hello,</p>
          <p style="font-size: 16px; color: #555555;">Use the code below to complete your sign-in process. This code is valid for 5 minutes.</p>
          
          <div style="text-align: center; margin: 30px 0;">
            <span style="font-size: 32px; font-weight: bold; color: #2c3e50; letter-spacing: 5px; background-color: #e8f0fe; padding: 10px 20px; border-radius: 5px; border: 1px dashed #2c3e50;">
              {otp_code}
            </span>
          </div>

          <p style="font-size: 14px; color: #888888; text-align: center;">
            If you didn't request this, you can safely ignore this email.
          </p>
          
          <hr style="border: 0; border-top: 1px solid #eeeeee; margin-top: 30px;">
          <div style="text-align: center; margin-top: 20px;">
            <img src="https://placehold.co/80x80?text=SoulJunction" alt="Soul Junction Logo" style="width: 80px; height: auto; opacity: 0.8;">
            <p style="font-size: 12px; color: #aaaaaa; margin-top: 5px;">&copy; 2024 Soul Junction Inc. All rights reserved.</p>
          </div>
          
        </div>
      </body>
    </html>
    """

    # 3. Attach the HTML to the email
    part_html = MIMEText(html_content, "html")
    msg.attach(part_html)

    try:
        # UPDATED: Use Namecheap's server settings (SSL Port 465)
        with smtplib.SMTP_SSL('mail.souljunction.life', 465) as server:
            server.login(NOREPLY_EMAIL, NOREPLY_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending Namecheap email: {e}")
        return False

# --- Controller 1: Send OTP ---
def send_otp_controller():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({"error": "Email is required"}), 400

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    
    # Store OTP
    otp_storage[email] = otp

    # Send Email
    if send_email_otp(email, otp):
        return jsonify({"success": True, "message": f"OTP sent to {email}"}), 200
    else:
        return jsonify({"success": False, "message": "Failed to send email"}), 500

# --- Controller 2: Verify OTP ---
def verify_otp_controller():
    data = request.get_json()
    email = data.get('email')
    user_otp = data.get('otp')

    if not email or not user_otp:
        return jsonify({"error": "Email and OTP are required"}), 400

    stored_otp = otp_storage.get(email)

    if not stored_otp:
        return jsonify({"error": "No OTP found or expired"}), 400

    if stored_otp == user_otp:
        del otp_storage[email]  # Clear OTP after success
        return jsonify({"success": True, "message": "Verification Successful!"}), 200
    else:
        return jsonify({"error": "Invalid OTP"}), 400
