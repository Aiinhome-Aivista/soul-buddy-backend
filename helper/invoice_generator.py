from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from datetime import datetime
import os


def generate_invoice_pdf(invoice_data):
    os.makedirs("invoices", exist_ok=True)

    filename = f"invoices/{invoice_data['invoice_number']}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)

    width, height = A4

    PRIMARY = HexColor("#111827")
    SECONDARY = HexColor("#6B7280")
    ACCENT = HexColor("#16A34A")
    LIGHT_BG = HexColor("#F3F4F6")

    # ================= HEADER =================
    c.setFillColor(PRIMARY)
    c.rect(0, height - 110, width, 110, fill=1, stroke=0)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, height - 60, "SOULJUNCTION")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 85, "support@souljunction.com")
    c.drawString(40, height - 100, "https://souljunction.life")

    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(width - 40, height - 60, "INVOICE")

    c.setFont("Helvetica", 10)
    c.drawRightString(width - 40, height - 80, f"Invoice No: {invoice_data['invoice_number']}")
    c.drawRightString(width - 40, height - 95, f"Date: {invoice_data['date']}")

    # ================= BILLED TO =================
    y = height - 160
    c.setFillColor(PRIMARY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Billed To")

    y -= 18

    # 🔹 USER NAME
    if invoice_data.get("full_name"):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, invoice_data["full_name"])
        y -= 14

    # 🔹 ADDRESS LINES
    c.setFont("Helvetica", 10)
    for line in invoice_data.get("billing_lines", []):
        if line:
            c.drawString(40, y, line)
            y -= 14

    # ================= TABLE HEADER =================
    y -= 25
    c.setFillColor(LIGHT_BG)
    c.rect(40, y, width - 80, 28, fill=1, stroke=0)

    c.setFillColor(PRIMARY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(45, y + 9, "Plan")
    c.drawString(230, y + 9, "Validity")
    c.drawString(370, y + 9, "Payment Method")
    c.drawRightString(width - 45, y + 9, "Amount")

    # ================= TABLE ROW =================
    y -= 28
    c.setFont("Helvetica", 10)
    c.drawString(45, y + 9, invoice_data["plan_name"])
    c.drawString(230, y + 9, f"{invoice_data['start_date']} → {invoice_data['end_date']}")
    c.drawString(370, y + 9, invoice_data["payment_method"])
    c.drawRightString(width - 45, y + 9, f"{invoice_data['currency']} {float(invoice_data['amount']):.2f}")

    # ================= SUMMARY =================
    y -= 50
    right_x = width - 45

    def draw_summary(label, value, y_pos, bold=False):
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 10)
        c.setFillColor(PRIMARY if bold else SECONDARY)
        c.drawRightString(right_x - 120, y_pos, label)
        c.setFillColor(PRIMARY)
        c.drawRightString(right_x, y_pos, value)

    amount = float(invoice_data["amount"])
    discount = float(invoice_data.get("discount_amount") or 0)
    final_amount = float(invoice_data.get("final_amount") or (amount - discount))

    # Subtotal
    draw_summary("Subtotal", f"{invoice_data['currency']} {amount:.2f}", y)
    y -= 18

    # Discount (only if > 0)
    if discount > 0:
        draw_summary("Discount", f"- {invoice_data['currency']} {discount:.2f}", y)
        y -= 18

        # Coupon code label
        if invoice_data.get("coupon_code"):
            c.setFont("Helvetica", 9)
            c.setFillColor(SECONDARY)
            c.drawRightString(
                right_x,
                y,
                f"Coupon Applied: {invoice_data['coupon_code']}"
            )
            y -= 14

    # Tax
    draw_summary("Tax (0%)", f"{invoice_data['currency']} 0.00", y)
    y -= 25

    # Total Paid
    draw_summary("Total Paid", f"{invoice_data['currency']} {final_amount:.2f}", y, bold=True)

    # ================= PAID BADGE =================
    c.setFillColor(ACCENT)
    c.roundRect(40, y - 10, 90, 28, 14, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(85, y - 2, "PAID")

    # ================= FOOTER =================
    c.setFillColor(SECONDARY)
    c.setFont("Helvetica", 9)
    c.drawString(40, 60, "Thank you for your purchase with Soul Junction.")
    c.drawString(40, 45, "For support: support@souljunction.com")
    c.drawString(40, 30, "This is a system-generated invoice.")

    c.save()
    return filename
