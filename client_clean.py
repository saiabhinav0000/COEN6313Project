import requests
import json
from word2number import w2n

BASE_URL = "http://127.0.0.1:8080"
#BASE_URL = "https://banking-server-320930579701.northamerica-northeast1.run.app"
#BASE_URL = "http://localhost:5000"

def print_response(response):
    print("Status Code:", response.status_code)
    try:
        print("Response:", response.json())
    except json.JSONDecodeError:
        print("Response:", response.text)
    print("-" * 50)

def test_register():
    url = f"{BASE_URL}/register"
    payload = {
        "name": "Test1",
        "email": "test@example.com",
        "password": "securepassword"
    }
    files = {
    "signature": open("IMG/1.jpg", "rb")  # Replace with a valid signature file path
    }
    response = requests.post(url, data=payload, files=files)
    print_response(response)
    return response.json().get("account_id")

def test_login():
    url = f"{BASE_URL}/login"
    payload = {
        "email": "test@example.com",
        "password": "securepassword"
    }
    print("Testing /login API")
    response = requests.post(url, json=payload)
    print_response(response)
    return response.json().get("token")

def test_check_balance(token):
    url = f"{BASE_URL}/balance"
    headers = {"Authorization": token}
    print("Testing /balance API")
    response = requests.get(url, headers=headers)
    print_response(response)

def test_deposit(token, amount):
    url = f"{BASE_URL}/deposit"
    headers = {"Authorization": token}
    payload = {"amount": amount}
    print("Testing /deposit API")
    response = requests.post(url, headers=headers, json=payload)
    print_response(response)

def test_withdraw(token, amount):
    url = f"{BASE_URL}/withdraw"
    headers = {"Authorization": token}
    payload = {"amount": amount}
    print("Testing /withdraw API")
    response = requests.post(url, headers=headers, json=payload)
    print_response(response)

def test_transfer(token, amount, to_account_email):
    url = f"{BASE_URL}/transfer"
    headers = {"Authorization": token}
    payload = {
        "amount": amount,
        "to_account_email": to_account_email
    }
    print("Testing /transfer API")
    response = requests.post(url, headers=headers, json=payload)
    print_response(response)

def test_transaction_history(token):
    url = f"{BASE_URL}/transaction_history"
    headers = {"Authorization": token}
    print("Testing /transaction_history API")
    response = requests.get(url, headers=headers)
    print_response(response)

def test_create_ticket(token, issue_description):
    url = f"{BASE_URL}/chatbot"
    headers = {"Authorization": token}
    payload = {"message": f"create a ticket: {issue_description}"}
    print("Testing /chatbot API - Create Ticket")
    response = requests.post(url, headers=headers, json=payload)
    print_response(response)

def test_view_tickets(token):
    url = f"{BASE_URL}/chatbot"
    headers = {"Authorization": token}
    payload = {"message": "view my tickets"}
    print("Testing /chatbot API - View Tickets")
    response = requests.post(url, headers=headers, json=payload)
    print_response(response)

def test_shop_items(token):
    url = f"{BASE_URL}/shop"
    headers = {"Authorization": token}
    print("Testing /shop API")
    response = requests.get(url, headers=headers)
    print_response(response)

def test_purchase(token, item_id, quantity):
    url = f"{BASE_URL}/purchase"
    headers = {"Authorization": token}
    payload = {
        "item_id": item_id,
        "quantity": quantity
    }
    print("Testing /purchase API")
    response = requests.post(url, headers=headers, json=payload)
    print_response(response)

def test_purchase_history(token):
    url = f"{BASE_URL}/purchase_history"
    headers = {"Authorization": token}
    print("Testing /purchase_history API")
    response = requests.get(url, headers=headers)
    print_response(response)
        
def test_process_cheque(token):
    url = f"{BASE_URL}/process_cheque"
    headers = {"Authorization": token}
    
    cheque_image_path = "t1.jpg"
    signature_image_path = "1.jpg"
    
    print("Testing /process_cheque API")
    with open(cheque_image_path, "rb") as cheque_file, open(signature_image_path, "rb") as signature_file:
        files = {
            "cheque": cheque_file,
            "signature": signature_file
        }
        response = requests.post(url, headers=headers, files=files)
    
    print_response(response)        

def run_tests():
    print("Starting API tests...")
    """
    account_id = test_register()  # Step 1
    if not account_id:
        print("Registration failed. Cannot proceed with tests.")
        return
    """
    token = test_login()
    if not token:
        print("Login failed. Cannot proceed with tests.")
        return

    test_check_balance(token)
    test_deposit(token, 500)
    test_withdraw(token, 100)
    test_transaction_history(token)

    test_transfer(token, 50, "alice@example.com")
    test_create_ticket(token, "This is a test issue.")
    test_view_tickets(token)
    test_purchase_history(token)
    test_process_cheque(token)

if __name__ == "__main__":
    run_tests()