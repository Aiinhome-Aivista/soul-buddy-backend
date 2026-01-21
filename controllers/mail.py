# import smtplib
# from email.mime.text import MIMEText
# from flask import request, jsonify
# from database.config import GMAIL_USER, GMAIL_APP_PASSWORD, RECEIVER_EMAIL


# def send_email(name, user_email, subject, message_body):
#     """Sends an email using Gmail SMTP."""
#     msg_content = f"Name: {name}\nEmail: {user_email}\nSubject: {subject}\n\nMessage:\n{message_body}"
#     msg = MIMEText(msg_content)
#     msg['Subject'] = f"New Contact Form: {subject}"
#     msg['From'] = GMAIL_USER
#     msg['To'] = RECEIVER_EMAIL

#     try:
#         with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#             server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
#             server.send_message(msg)
#         return True
#     except Exception as e:
#         print(f"Error sending email: {e}")
#         return False

# def handle_contact_controller():
#     data = request.json
#     name = data.get('name')
#     email = data.get('email')
#     subject = data.get('subject')
#     message = data.get('message')

#     if not all([name, email, subject, message]):
#         return jsonify({"error": "Missing fields"}), 400

#     if send_email(name, email, subject, message):
#         return jsonify({"success": True, "message": "Email sent!"}), 200
#     else:
#         return jsonify({"success": False, "message": "Failed to send email"}), 500



# import smtplib
# from email.mime.text import MIMEText
# from flask import request, jsonify

# # Update imports to use your Namecheap config variables
# from database.config import SUPPORT_EMAIL, SUPPORT_PASSWORD, RECEIVER_EMAIL

# def send_email(name, user_email, subject, message_body):
#     """
#     Sends a contact form email using Namecheap SMTP.
#     """
    
#     # We include the user's details inside the email body
#     msg_content = f"Name: {name}\nUser Email: {user_email}\nSubject: {subject}\n\nMessage:\n{message_body}"
    
#     msg = MIMEText(msg_content)
#     msg['Subject'] = f"New Contact Form: {subject}"
    
#     # IMPORTANT: The 'From' address must be your authenticated Namecheap email
#     msg['From'] = SUPPORT_EMAIL 
#     msg['To'] = RECEIVER_EMAIL
    
#     # Optional: Add Reply-To so you can hit "Reply" and it goes to the user, not support
#     msg.add_header('Reply-To', user_email)

#     try:
#         # UPDATED: Namecheap SMTP Settings
#         # Server: mail.privateemail.com
#         # Port: 465 (SSL)
#         with smtplib.SMTP_SSL('mail.souljunction.life', 465) as server:
#             server.login(SUPPORT_EMAIL, SUPPORT_PASSWORD)
#             server.send_message(msg)
#         return True
#     except Exception as e:
#         print(f"Error sending Namecheap email: {e}")
#         return False

# def handle_contact_controller():
#     data = request.json
#     name = data.get('name')
#     email = data.get('email')
#     subject = data.get('subject')
#     message = data.get('message')

#     if not all([name, email, subject, message]):
#         return jsonify({"error": "Missing fields"}), 400

#     if send_email(name, email, subject, message):
#         return jsonify({"success": True, "message": "Email sent!"}), 200
#     else:
#         return jsonify({"success": False, "message": "Failed to send email"}), 500



import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart # Added this import
from flask import request, jsonify

# Update imports to use your Namecheap config variables
from database.config import SUPPORT_EMAIL, SUPPORT_PASSWORD, RECEIVER_EMAIL

def send_email(name, user_email, subject, message_body):
    """
    Sends a professional HTML contact form notification to the Admin.
    """
    
    try:
        # 1. Setup the Email Container
        msg = MIMEMultipart("alternative")
        msg['Subject'] = f"Soul Junction Inquiry: {subject}"
        msg['From'] = SUPPORT_EMAIL 
        msg['To'] = RECEIVER_EMAIL
        msg.add_header('Reply-To', user_email)

        # 2. Plain Text Fallback (For summaries/notifications)
        text_body = f"""
        New Contact Form Submission
        
        From: {name} ({user_email})
        Subject: {subject}
        
        Message:
        {message_body}
        """

        # 3. Professional HTML Design (Admin View)
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <style>
            body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f6f8; padding: 20px; }}
            .card {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); overflow: hidden; border-left: 5px solid #0078d4; }}
            .header {{ background-color: #ffffff; padding: 25px 30px; border-bottom: 1px solid #eeeeee; }}
            .brand {{ color: #0078d4; font-size: 14px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; display: block; }}
            .title {{ margin: 0; color: #333; font-size: 20px; font-weight: 600; }}
            .meta {{ padding: 20px 30px; background-color: #fafbfc; font-size: 14px; color: #555; }}
            .meta p {{ margin: 5px 0; }}
            .message-box {{ padding: 30px; font-size: 16px; color: #333; line-height: 1.6; }}
            .footer {{ padding: 15px 30px; background-color: #f4f6f8; text-align: right; }}
            .button {{ background-color: #0078d4; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: bold; font-size: 14px; }}
          </style>
        </head>
        <body>
          <div class="card">
            
            <div class="header">
              <span class="brand">Soul Junction Website</span>
              <h2 class="title">New User Inquiry</h2>
            </div>

            <div class="meta">
              <p><strong>From:</strong> {name}</p>
              <p><strong>Email:</strong> <a href="mailto:{user_email}" style="color: #0078d4;">{user_email}</a></p>
              <p><strong>Subject:</strong> {subject}</p>
            </div>

            <div class="message-box">
              {message_body}
            </div>

            <div class="footer">
               <a href="mailto:{user_email}?subject=Re: {subject}" class="button">Reply to User</a>
            </div>

          </div>
        </body>
        </html>
        """

        # 4. Attach Parts
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # 5. Send using Namecheap SMTP
        # Note: Ensure 'mail.souljunction.life' is the correct server alias 
        # provided by Namecheap. Usually it is 'mail.privateemail.com'.
        with smtplib.SMTP_SSL('mail.souljunction.life', 465) as server:
            server.login(SUPPORT_EMAIL, SUPPORT_PASSWORD)
            server.send_message(msg)
        return True

    except Exception as e:
        print(f"Error sending Namecheap email: {e}")
        return False

def handle_contact_controller():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')

    if not all([name, email, subject, message]):
        return jsonify({"error": "Missing fields"}), 400

    if send_email(name, email, subject, message):
        return jsonify({"success": True, "message": "Email sent!"}), 200
    else:
        return jsonify({"success": False, "message": "Failed to send email"}), 500