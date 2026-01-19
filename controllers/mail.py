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



import smtplib
from email.mime.text import MIMEText
from flask import request, jsonify

# Update imports to use your Namecheap config variables
from database.config import SUPPORT_EMAIL, SUPPORT_PASSWORD, RECEIVER_EMAIL

def send_email(name, user_email, subject, message_body):
    """
    Sends a contact form email using Namecheap SMTP.
    """
    
    # We include the user's details inside the email body
    msg_content = f"Name: {name}\nUser Email: {user_email}\nSubject: {subject}\n\nMessage:\n{message_body}"
    
    msg = MIMEText(msg_content)
    msg['Subject'] = f"New Contact Form: {subject}"
    
    # IMPORTANT: The 'From' address must be your authenticated Namecheap email
    msg['From'] = SUPPORT_EMAIL 
    msg['To'] = RECEIVER_EMAIL
    
    # Optional: Add Reply-To so you can hit "Reply" and it goes to the user, not support
    msg.add_header('Reply-To', user_email)

    try:
        # UPDATED: Namecheap SMTP Settings
        # Server: mail.privateemail.com
        # Port: 465 (SSL)
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