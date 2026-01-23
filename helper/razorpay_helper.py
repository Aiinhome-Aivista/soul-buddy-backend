import razorpay
from database.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET

# ------------------ Razorpay Client ------------------
razorpay_client = razorpay.Client(
    auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
)

# ------------------ Currency Multipliers ------------------
# Razorpay expects the smallest unit of currency
# INR -> paise, USD -> cents, EUR -> cents, etc.
CURRENCY_MULTIPLIER = {
    "INR": 100,   # paise
    "USD": 100,   # cents
    "EUR": 100,
    "GBP": 100
}

SUPPORTED_CURRENCIES = ["INR", "USD"]

# ------------------ Create Razorpay Order ------------------
def create_razorpay_order(amount, currency="USD"):
    """
    Creates a Razorpay order for the given amount and currency.

    :param amount: int or float (e.g. 9.99 or 10)
    :param currency: "USD" or "INR"
    :return: Razorpay order dict
    """

    if not amount or amount <= 0:
        raise ValueError("Amount must be greater than 0")

    currency = currency.upper()

    if currency not in SUPPORTED_CURRENCIES:
        raise ValueError(f"Unsupported currency: {currency}")

    multiplier = CURRENCY_MULTIPLIER.get(currency, 100)

    # Convert to smallest unit (cents/paise)
    razorpay_amount = int(round(float(amount) * multiplier))

    order_data = {
        "amount": razorpay_amount,
        "currency": currency,
        "payment_capture": 1
    }

    order = razorpay_client.order.create(order_data)
    return order
