from flask import Flask, jsonify, request
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
from bson.errors import InvalidId
import jwt as pyjwt
from datetime import datetime, timedelta
import os
import re
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel, AutoModelForCausalLM, AutoTokenizer
from google.cloud import vision
import io
from word2number import w2n
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'E5EEEEEEEEE'  # Replace with a secure key
CORS(app)  # This will allow all origins (for development).

client = MongoClient("mongodb+srv://ashrafuddinrafat:zspkkdmFMioU3vEa@cluster0.pq8jro5.mongodb.net/banking_system?retryWrites=true&w=majority")
db = client['banking_system']
tickets_collection = db['tickets']          # New collection for tickets
accounts_collection = db['accounts']
shop_collection = db['shop']
transactions_collection = db['transactions']
purchase_information_collection = db['purchase_information']

print("Loading DialoGPT-medium model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
print("DialoGPT-medium model loaded successfully.")

model_name = "microsoft/swin-tiny-patch4-window7-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
signature_model = AutoModel.from_pretrained(model_name)
signature_model.eval()

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\Abhi\COEN6313 Proj\coen6313proj-442020-2cc291d18994.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/app/credentials/coen6313proj-442020-2cc291d18994.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

conversation_history = {}

INTENTS = [
    "transfer_money",
    "check_balance",
    "view_transaction_history",
    "create_ticket",          # Existing intent
    "view_tickets",           # New intent
    "greeting",
    "goodbye",
    "unknown"
]

def extract_signature_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = signature_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def verify_signature(signature1_path, signature2_path, threshold=0.8):
    embedding1 = extract_signature_features(signature1_path)
    embedding2 = extract_signature_features(signature2_path)
    similarity = F.cosine_similarity(embedding1, embedding2).item()
    return similarity >= threshold

def extract_text_from_cheque(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print('No text found in image.')
        return None, None

    full_text = texts[0].description
    print('Detected text:', full_text)

    lines = full_text.split('\n')

    ignored_lines = [
        "PAY TO THE",
        "ORDER OF",
        "MEMO",
        "DATE",
        "$",
        "5679",
        "DOLLARS",
        "⑈325760408⑈",
        "003192⑆",
        "42",
        "23",
        "24",
        "69",
        "003192⑆ 0583",
        "⑈325760408⑈ 003192⑆ 0583 42"
    ]

    filtered_lines = [line.strip() for line in lines if line.strip() not in ignored_lines]

    print(f"Filtered Lines: {filtered_lines}")  # Debugging

    if len(filtered_lines) < 2:
        print("Not enough lines after filtering to extract name and amounts.")
        return None, None

    name = filtered_lines[0]
    print(f"Extracted Name: {name}")

    numeric_amount = None
    written_amount = None

    if len(filtered_lines) >= 3:
        if filtered_lines[1].isdigit():  # Check if second line is numeric
            numeric_amount = filtered_lines[1]
            written_amount = filtered_lines[2]  # Third line would be written amount
        elif filtered_lines[2].isdigit():  # Check if third line is numeric
            numeric_amount = filtered_lines[2]
            written_amount = filtered_lines[1]  # Second line would be written amount
    else:
        numeric_amount = filtered_lines[1]

    print(f"Extracted Written Amount: {written_amount}")
    print(f"Extracted Numeric Amount: {numeric_amount}")

    try:
        converted_amount = w2n.word_to_num(written_amount)
        print(f"Converted Written Amount: {converted_amount}")
    except ValueError as e:
        print(f"Failed to convert written amount '{written_amount}': {e}")
        return None, None

    try:
        if int(numeric_amount) == converted_amount:
            print(f"Matched Numeric Amount: {numeric_amount}")
            print(f"Matched Written Amount: {written_amount}")
            return name, numeric_amount
        else:
            print("Numeric and written amounts do not match or could not be validated.")
            return None, None
    except ValueError:
        print(f"Failed to convert numeric amount '{numeric_amount}' to integer.")
        return None, None


def get_account(identifier):
    if isinstance(identifier, ObjectId):
        return accounts_collection.find_one({"_id": identifier})
    elif isinstance(identifier, str):
        try:
            obj_id = ObjectId(identifier)
            account = accounts_collection.find_one({"_id": obj_id})
            if account:
                return account
        except InvalidId:
            pass
        return accounts_collection.find_one({"email": identifier})
    else:
        return None

def token_required(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 403
        try:
            data = pyjwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = get_account(data['account_id'])
            if not current_user:
                return jsonify({'error': 'User not found!'}), 403
        except pyjwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired!'}), 403
        except pyjwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token!'}), 403
        return f(current_user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper
    
# Custom Layer for Positional Encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model)
        # Apply sin to even indices in the array; 2i
        sines = np.sin(angle_rads[:, 0::2])
        # Apply cos to odd indices in the array; 2i+1
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Encoder Layer
def transformer_encoder(inputs, num_heads, ff_dim, dropout=0.1):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=inputs.shape[-1], num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res  # Another residual connection

# Build Transformer Model
def build_model(input_window, output_window, d_model=64, num_heads=2, ff_dim=128, dropout=0.1):
    inputs = keras.Input(shape=(input_window, 1))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(sequence_length=input_window, d_model=d_model)(x)
    x = transformer_encoder(x, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(output_window * 1)(x)
    outputs = layers.Reshape((output_window, 1))(x)
    model = keras.Model(inputs, outputs)
    return model

@app.route('/register', methods=['POST'])
def register():
    data = request.form
    name = data.get("name")
    email = data.get("email")
    password = data.get("password").encode('utf-8')
    signature_file = request.files.get("signature")

    if accounts_collection.find_one({"email": email}):
        return jsonify({'error': 'Email already exists!'}), 400

    hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
    signature_path = f"signatures/{email}_signature.jpg"
    signature_file.save(signature_path)

    new_account = {
        "name": name,
        "email": email,
        "password": hashed_password,
        "balance": 0,
        "created_at": datetime.utcnow(),
        "signature_path": signature_path
    }
    result = accounts_collection.insert_one(new_account)
    return jsonify({'message': 'Account registered successfully!', 'account_id': str(result.inserted_id)}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password").encode('utf-8')
    
    user = accounts_collection.find_one({"email": email})
    if not user or not bcrypt.checkpw(password, user['password']):
        return jsonify({'error': 'Invalid credentials!'}), 401

    token = pyjwt.encode({
        'account_id': str(user['_id']),
        'exp': datetime.utcnow() + timedelta(hours=1)
    }, app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'token': token}), 200
    
@app.route('/predict-expenses', methods=['GET'])
def predict_expenses():
    account_id = request.args.get('account_id')
    if not account_id:
        return jsonify({'error': 'account_id parameter is required'}), 400

    # Fetch purchase data for the given account_id
    purchases = list(purchase_information_collection.find({'account_id': account_id}))

    if not purchases:
        return jsonify({'error': 'No purchase data found for the given account_id'}), 404

    # Convert data to DataFrame
    data = pd.DataFrame(purchases)

    # Ensure timestamp field is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Check if 'category' column exists
    if 'category' not in data.columns:
        return jsonify({'error': 'No category data found in the purchase information'}), 400

    # Group data by category and month and sum the total_price
    data['month'] = data['timestamp'].dt.to_period('M')
    grouped = data.groupby(['category', 'month']).agg({'total_price': 'sum'}).reset_index()

    # Prepare predictions per category
    predictions_dict = {}
    total_predictions = None  # To store total predicted expenses per month
    future_dates = None  # To store future dates for which predictions are made

    for category in grouped['category'].unique():
        category_data = grouped[grouped['category'] == category]
        # Ensure there are enough data points
        if len(category_data) < 12:
            # Skip categories with less than 12 months of data
            continue

        # Sort the data by 'month'
        category_data = category_data.sort_values('month')

        # Convert 'month' to datetime
        category_data['month'] = category_data['month'].dt.to_timestamp()

        # Get the prices as numpy array
        prices = category_data['total_price'].values.astype('float32')

        # Define input and output window sizes
        input_window = 6
        output_window = 3
        sequence_length = input_window + output_window

        # Check if we have enough data points
        if len(prices) < sequence_length:
            # Skip categories with insufficient data
            continue

        # Create sequences
        def create_sequences(data, input_window, output_window):
            sequences = []
            for i in range(len(data) - sequence_length + 1):
                input_seq = data[i:i+input_window]
                target_seq = data[i+input_window:i+sequence_length]
                sequences.append((input_seq, target_seq))
            return sequences

        sequences = create_sequences(prices, input_window, output_window)
        if not sequences:
            continue  # Skip if no sequences can be created

        inputs = np.array([s[0] for s in sequences])
        targets = np.array([s[1] for s in sequences])

        # Reshape inputs and targets
        inputs = inputs[..., np.newaxis]  # Shape: (num_samples, input_window, 1)
        targets = targets[..., np.newaxis]  # Shape: (num_samples, output_window, 1)

        # Build the model
        model = build_model(input_window, output_window)

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Train the model
        model.fit(inputs, targets, epochs=50, batch_size=4, verbose=0)

        # Prepare the last input sequence for prediction
        last_input = prices[-input_window:]
        last_input = last_input[np.newaxis, ..., np.newaxis]  # Shape: (1, input_window, 1)

        # Predict the next output_window values
        prediction = model.predict(last_input)
        predicted_values = prediction[0, :, 0]

        # Prepare future dates
        last_date = category_data['month'].max()
        future_dates_category = pd.date_range(start=last_date + pd.offsets.MonthBegin(), periods=output_window, freq='MS')

        # Store future dates (they are the same for all categories)
        if future_dates is None:
            future_dates = future_dates_category

        # Prepare the predictions for this category
        category_predictions = {date.strftime('%Y-%m'): float(value) for date, value in zip(future_dates_category, predicted_values)}

        # Add to the main predictions dictionary
        predictions_dict[category] = category_predictions

        # Accumulate total predictions
        if total_predictions is None:
            # Initialize total_predictions with zeros
            total_predictions = pd.Series(0.0, index=future_dates_category)
        total_predictions += pd.Series(predicted_values, index=future_dates_category)

    if not predictions_dict:
        return jsonify({'error': 'Not enough data to make a prediction for any category (minimum 12 months required per category)'}), 400

    # Prepare total predictions dictionary
    total_predictions_dict = {date.strftime('%Y-%m'): float(value) for date, value in total_predictions.items()}

    # Prepare the response
    response = {
        'account_id': account_id,
        'predicted_expenses': predictions_dict,
        'total_predicted_expenses': total_predictions_dict
    }

    return jsonify(response)

@app.route('/actual-expenses', methods=['GET'])
def actual_expenses():
    account_id = request.args.get('account_id')
    if not account_id:
        return jsonify({'error': 'account_id parameter is required'}), 400

    # Fetch purchase data for the given account_id
    purchases = list(purchase_information_collection.find({'account_id': account_id}))

    if not purchases:
        return jsonify({'error': 'No purchase data found for the given account_id'}), 404

    # Convert data to DataFrame
    data = pd.DataFrame(purchases)

    # Ensure 'timestamp' and 'category' fields exist
    if 'timestamp' not in data.columns or 'category' not in data.columns:
        return jsonify({'error': 'Required fields are missing in the purchase data'}), 400

    # Ensure 'timestamp' field is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Group data by category and month and sum the total_price
    data['month'] = data['timestamp'].dt.to_period('M')
    grouped = data.groupby(['category', 'month']).agg({'total_price': 'sum'}).reset_index()

    # Convert 'month' from Period to string for JSON serialization
    grouped['month'] = grouped['month'].astype(str)

    # Prepare the response
    actual_expenses_dict = {}
    for category in grouped['category'].unique():
        category_data = grouped[grouped['category'] == category]
        expenses = {row['month']: float(row['total_price']) for index, row in category_data.iterrows()}
        actual_expenses_dict[category] = expenses

    # Calculate total expenses per month across all categories
    total_expenses = data.groupby('month').agg({'total_price': 'sum'}).reset_index()
    total_expenses['month'] = total_expenses['month'].astype(str)
    total_expenses_dict = {row['month']: float(row['total_price']) for index, row in total_expenses.iterrows()}

    response = {
        'account_id': account_id,
        'actual_expenses': actual_expenses_dict,
        'total_actual_expenses': total_expenses_dict
    }

    return jsonify(response)

    
@app.route('/process_cheque', methods=['POST'])
@token_required
def process_cheque(current_user):
    try:
        cheque_file = request.files.get("cheque")
        signature_file = request.files.get("signature")
        
        # Validate file inputs
        if not cheque_file or not signature_file:
            return jsonify({'error': 'Cheque and signature files are required!'}), 400

        # Save files temporarily
        cheque_path = "temp_cheque.jpg"
        signature_path = "temp_signature.jpg"
        cheque_file.save(cheque_path)
        signature_file.save(signature_path)

        # Extract data from the cheque
        name, amount = extract_text_from_cheque(cheque_path)
        if not name or not amount:
            os.remove(cheque_path)
            os.remove(signature_path)
            return jsonify({'error': 'Failed to extract data from the cheque.'}), 400

        # Validate the extracted name with the current user
        if name.lower() != current_user.get('name', '').lower():
            os.remove(cheque_path)
            os.remove(signature_path)
            return jsonify({'error': 'Name on the cheque does not match the account holder!'}), 400

        # Verify the signature
        stored_signature_path = current_user.get('signature_path')
        if not stored_signature_path or not os.path.exists(stored_signature_path):
            os.remove(cheque_path)
            os.remove(signature_path)
            return jsonify({'error': 'Reference signature not found!'}), 404

        match = verify_signature(stored_signature_path, signature_path)
        os.remove(cheque_path)
        os.remove(signature_path)

        if not match:
            return jsonify({'error': 'Signature does not match the reference!'}), 400

        # Update the account balance
        accounts_collection.update_one(
            {"_id": current_user['_id']},
            {"$inc": {"balance": float(amount)}}
        )

        # Record the transaction
        transaction = {
            "account_id": current_user['_id'],
            "type": "deposit",
            "amount": float(amount),
            "timestamp": datetime.utcnow()
        }
        transactions_collection.insert_one(transaction)

        # Get updated account information
        updated_account = get_account(current_user['_id'])

        return jsonify({
            'message': 'Cheque processed successfully!',
            'name': name,
            'amount': amount,
            'balance': updated_account['balance']
        }), 200

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

@app.route('/balance', methods=['GET'])
@token_required
def check_balance(current_user):
    account_id = str(current_user['_id'])
    account = get_account(account_id)
    return jsonify({'account_id': account_id, 'balance': account['balance']}), 200

@app.route('/deposit', methods=['POST'])
@token_required
def deposit(current_user):
    account_id = str(current_user['_id'])
    amount = request.json.get('amount')

    if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
        return jsonify({'error': 'Deposit amount must be a positive number'}), 400

    accounts_collection.update_one(
        {"_id": ObjectId(account_id)},
        {"$inc": {"balance": amount}}
    )

    transaction = {
        "account_id": ObjectId(account_id),
        "type": "deposit",
        "amount": amount,
        "timestamp": datetime.utcnow()
    }
    transactions_collection.insert_one(transaction)

    updated_account = get_account(account_id)
    return jsonify({'account_id': account_id, 'balance': updated_account['balance']}), 200

@app.route('/withdraw', methods=['POST'])
@token_required
def withdraw(current_user):
    account_id = str(current_user['_id'])
    amount = request.json.get('amount')

    if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
        return jsonify({'error': 'Withdrawal amount must be a positive number'}), 400

    account = get_account(account_id)
    if account['balance'] < amount:
        return jsonify({'error': 'Insufficient funds'}), 400

    accounts_collection.update_one(
        {"_id": ObjectId(account_id)},
        {"$inc": {"balance": -amount}}
    )

    transaction = {
        "account_id": ObjectId(account_id),
        "type": "withdrawal",
        "amount": amount,
        "timestamp": datetime.utcnow()
    }
    transactions_collection.insert_one(transaction)

    updated_account = get_account(account_id)
    return jsonify({'account_id': account_id, 'balance': updated_account['balance']}), 200

@app.route('/transfer', methods=['POST'])
@token_required
def transfer(current_user):
    from_account_id = str(current_user['_id'])
    amount = request.json.get('amount')
    to_account_email = request.json.get('to_account_email')

    if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
        return jsonify({'error': 'Transfer amount must be a positive number'}), 400

    to_account = get_account(to_account_email)
    if not to_account:
        return jsonify({'error': 'Recipient account not found'}), 404

    if current_user['balance'] < amount:
        return jsonify({'error': 'Insufficient funds'}), 400

    try:
        with client.start_session() as session:
            with session.start_transaction():
                accounts_collection.update_one({"_id": ObjectId(from_account_id)}, {"$inc": {"balance": -amount}}, session=session)
                accounts_collection.update_one({"_id": ObjectId(to_account['_id'])}, {"$inc": {"balance": amount}}, session=session)

                transaction_out = {
                    "account_id": ObjectId(from_account_id),
                    "type": "transfer_out",
                    "amount": amount,
                    "to_account_id": ObjectId(to_account['_id']),
                    "timestamp": datetime.utcnow()
                }
                transaction_in = {
                    "account_id": ObjectId(to_account['_id']),
                    "type": "transfer_in",
                    "amount": amount,
                    "from_account_id": ObjectId(from_account_id),
                    "timestamp": datetime.utcnow()
                }
                transactions_collection.insert_many([transaction_out, transaction_in], session=session)
        return jsonify({'message': 'Transfer successful'}), 200
    except Exception as e:
        print(f"Transfer error: {e}")
        return jsonify({'error': 'An error occurred during the transfer.'}), 500

@app.route('/transaction_history', methods=['GET'])
@token_required
def transaction_history(current_user):
    account_id = current_user['_id']  # Assuming current_user['_id'] is already an ObjectId

    # Fetch transactions for the current user's account, sorted by timestamp (descending)
    transactions = list(transactions_collection.find({"account_id": account_id}).sort("timestamp", -1))

    # Convert ObjectId and datetime fields to JSON serializable types
    for transaction in transactions:
        # Convert ObjectId fields to strings
        transaction['_id'] = str(transaction['_id'])
        transaction['account_id'] = str(transaction['account_id'])
        if "to_account_id" in transaction:
            transaction["to_account_id"] = str(transaction["to_account_id"])
        if "from_account_id" in transaction:
            transaction["from_account_id"] = str(transaction["from_account_id"])
        
        # Convert timestamp to ISO 8601 string
        if "timestamp" in transaction and isinstance(transaction['timestamp'], datetime):
            transaction['timestamp'] = transaction['timestamp'].isoformat()

    # Return JSON response with transactions or an empty list
    return jsonify({'transaction_history': transactions or []}), 200

@app.route('/chatbot', methods=['POST'])
@token_required
def chatbot(current_user):
    data = request.json
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'response': 'Please provide a message.'}), 400

    intent = detect_intent(message)

    if intent == "transfer_money":
        response = handle_transfer_money(current_user, message)
    elif intent == "check_balance":
        response = handle_check_balance(current_user)
    elif intent == "view_transaction_history":
        response = handle_view_transaction_history(current_user)
    elif intent == "create_ticket":
        response = handle_create_ticket(current_user, message)
    elif intent == "view_tickets":    # New intent handling
        response = handle_view_tickets(current_user)
    else:
        response = generate_response(current_user['_id'], message)

    return jsonify({'response': response}), 200

def detect_intent(message):
    message = message.lower()
    if "transfer" in message and "to" in message:
        return "transfer_money"
    elif "balance" in message:
        return "check_balance"
    elif "transaction history" in message or "transactions" in message:
        return "view_transaction_history"
    elif any(ticket_phrase in message for ticket_phrase in ["help me", "report a problem", "create a ticket", "i need assistance", "support"]):
        return "create_ticket"
    elif any(view_phrase in message for view_phrase in ["view my tickets", "show my tickets", "my support requests", "view my support tickets"]):
        return "view_tickets"     # New intent detection
    else:
        return "unknown"

def handle_transfer_money(current_user, message):
    transfer_pattern = r"transfer\s+\$?(\d+(?:\.\d+)?)\s+to\s+([\w\.-]+@[\w\.-]+\.\w+)"
    match = re.search(transfer_pattern, message, re.IGNORECASE)
    
    if not match:
        return "Please specify the amount and recipient's email. For example, 'Transfer $100 to recipient@example.com'."
    
    amount = float(match.group(1))
    to_email = match.group(2)
    
    if amount <= 0:
        return "Transfer amount must be a positive number."
    
    to_account = get_account(to_email)
    if not to_account:
        return "Recipient account not found."
    
    if current_user['balance'] < amount:
        return "Insufficient funds for this transfer."
    
    from_account_id = str(current_user['_id'])
    
    try:
        with client.start_session() as session:
            with session.start_transaction():
                accounts_collection.update_one({"_id": ObjectId(from_account_id)}, {"$inc": {"balance": -amount}}, session=session)
                accounts_collection.update_one({"_id": ObjectId(to_account['_id'])}, {"$inc": {"balance": amount}}, session=session)

                transaction_out = {
                    "account_id": ObjectId(from_account_id),
                    "type": "transfer_out",
                    "amount": amount,
                    "to_account_id": ObjectId(to_account['_id']),
                    "timestamp": datetime.utcnow()
                }
                transaction_in = {
                    "account_id": ObjectId(to_account['_id']),
                    "type": "transfer_in",
                    "amount": amount,
                    "from_account_id": ObjectId(from_account_id),
                    "timestamp": datetime.utcnow()
                }
                transactions_collection.insert_many([transaction_out, transaction_in], session=session)
        return f"Transferred ${amount} to {to_email} successfully."
    except Exception as e:
        print(f"Transfer error: {e}")
        return "An error occurred during the transfer."

def handle_check_balance(current_user):
    account_id = str(current_user['_id'])
    account = get_account(account_id)
    return f"Your current balance is ${account['balance']}."

def handle_view_transaction_history(current_user):
    account_id = str(current_user['_id'])
    transactions = list(transactions_collection.find({"account_id": ObjectId(account_id)}).sort("timestamp", -1))
    
    if not transactions:
        return "No transactions found."

    history = []
    for txn in transactions[:10]:  # Limit to last 10 transactions
        txn_type = txn['type'].replace('_', ' ').title()
        amount = txn['amount']
        timestamp = txn['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if txn_type in ['Transfer Out', 'Transfer In']:
            related_account_id = txn.get('to_account_id') if 'to_account_id' in txn else txn.get('from_account_id')
            related_account = get_account(str(related_account_id))
            related_email = related_account['email'] if related_account else 'Unknown'
            direction = 'to' if 'to_account_id' in txn else 'from'
            history.append(f"{txn_type} of ${amount} {direction} {related_email} on {timestamp}")
        else:
            history.append(f"{txn_type} of ${amount} on {timestamp}")

    response_message = "Here are your recent transactions:\n\n" + "\n".join(history)
    return response_message

def handle_create_ticket(current_user, message):
    issue_pattern = r"create a ticket[:\-]?\s*(.*)"
    match = re.search(issue_pattern, message, re.IGNORECASE)
    
    if match:
        issue_description = match.group(1).strip()
    else:
        return "Sure, I can help you create a support ticket. Please describe your issue."
    
    if not issue_description:
        return "Please provide a detailed description of your issue to create a support ticket."
    
    ticket = {
        "user_id": ObjectId(current_user['_id']),
        "issue_description": issue_description,
        "status": "open",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    try:
        result = tickets_collection.insert_one(ticket)
        ticket_id = str(result.inserted_id)
        return f"Your support ticket has been created successfully! The next available agent will directly call you to support you or you can contact center. Ticket ID: {ticket_id}"
    except Exception as e:
        print(f"Ticket creation error: {e}")
        return "An error occurred while creating your support ticket. Please try again later."

def handle_view_tickets(current_user):
    account_id = str(current_user['_id'])
    
    tickets = list(tickets_collection.find({"user_id": ObjectId(account_id)}).sort("created_at", -1))
    
    if not tickets:
        return "You have no support tickets."
    
    ticket_messages = []
    for ticket in tickets[:10]:  # Limit to last 10 tickets
        ticket_id = str(ticket['_id'])
        issue = ticket.get('issue_description', 'No description provided.')
        status = ticket.get('status', 'N/A').title()
        created_at = ticket.get('created_at').strftime('%Y-%m-%d %H:%M:%S')
        ticket_messages.append(f"**Ticket ID:** {ticket_id}\n**Issue:** {issue}\n**Status:** {status}\n**Created At:** {created_at}\n")
    
    response_message = "Here are your recent support tickets:\n\n" + "\n".join(ticket_messages)
    return response_message

def handle_greetings(intent):
    if intent == "greeting":
        return "Hello! How can I assist you today?"
    elif intent == "goodbye":
        return "Goodbye! Have a great day!"
    else:
        return "Hello!"
    
@app.route('/shop', methods=['GET'])
@token_required

def shop_items(current_user):
    try:
        items = list(shop_collection.find())
        for item in items:
            item['_id'] = str(item['_id'])
            item['shop_name'] = item.get('Shop Name')
            item['category'] = item.get('Product Category')
            item['name'] = item.get('Product Name')
            item['price'] = item.get('Price')
        return jsonify({'items': items}), 200
        
    except Exception as e:
        
        return jsonify({'error': 'An unexpected error occurred while retrieving shop items.'}), 500
@app.route('/purchase', methods=['POST'])
@token_required

def purchase(current_user):
    try:
        data = request.json
        item_id = data.get("item_id")
        quantity = data.get("quantity", 1)  # Default quantity to 1 if not specified

        if not item_id or quantity <= 0:
            
            return jsonify({'error': 'Invalid item ID or quantity'}), 400

        try:
            item = db['shop'].find_one({"_id": ObjectId(item_id)})
        except InvalidId:
            
            return jsonify({'error': 'Invalid item ID format'}), 400

        if not item:
            
            return jsonify({'error': 'Item not found in shop'}), 404

       

        item_name = item.get('Product Name')  # Adjusted to match your field name
        if not item_name:
            
            item_name = "Unknown Item"
        
        shop_name = item.get('Shop Name')  # Adjusted to match your field name
        if not shop_name:
            
            item_name = "Unknown Item"
        
        product_category = item.get('Product Category')  # Adjusted to match your field name
        if not product_category:
           
            item_name = "Unknown Item"

        total_price = item.get('Price', 0.0) * quantity  # Adjusted to match your field name

        if current_user.get('balance', 0.0) < total_price:
            
            return jsonify({'error': 'Insufficient funds for this purchase'}), 400

        update_result = accounts_collection.update_one(
            {"_id": current_user['_id']},  # Correct usage without ObjectId() wrapper
            {"$inc": {"balance": -total_price}}
        )

        

        if update_result.matched_count == 0:
           
            return jsonify({'error': 'Failed to update balance'}), 500

        if update_result.modified_count == 0:
           
            return jsonify({'error': 'Failed to update balance'}), 500

        purchase_transaction = {
            "account_id": ObjectId(current_user['_id']),
            "type": "purchase",
            "item_id": ObjectId(item_id),
            "item_name": item_name,
            "shop_name": shop_name,
            "product_category": product_category,
            "quantity": quantity,
            "total_price": total_price,
            "timestamp": datetime.utcnow()
        }
        transactions_collection.insert_one(purchase_transaction)
        

        updated_account = get_account(current_user['_id'])

        return jsonify({
            'message': f'Successfully purchased {quantity} of {item_name}',
            'remaining_balance': updated_account.get('balance', 0.0)
        }), 200
    except Exception as e:
        
        return jsonify({'error': 'An unexpected error occurred during purchase.'}), 500
    
@app.route('/purchase_history', methods=['GET'])
@token_required
def purchase_history(current_user):
    try:
        account_id = current_user['_id']
        purchases = list(transactions_collection.find({
            "account_id": ObjectId(account_id),
            "type": "purchase"
        }).sort("timestamp", -1))

        for purchase in purchases:
            purchase['_id'] = str(purchase['_id'])
            purchase['account_id'] = str(purchase['account_id'])
            purchase['item_id'] = str(purchase['item_id'])
            purchase['timestamp'] = purchase['timestamp'].isoformat()

        

        return jsonify({'purchase_history': purchases}), 200
    except Exception as e:
        
        return jsonify({'error': 'An unexpected error occurred while retrieving purchase history.'}), 500


def generate_response(user_id, user_message):
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)
    conversation_history[user_id].append(new_user_input_ids)
    
    bot_input_ids = torch.cat(conversation_history[user_id], dim=-1)
    
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )
    
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    conversation_history[user_id].append(chat_history_ids[:, bot_input_ids.shape[-1]:])
    
    if len(conversation_history[user_id]) > 10:
        conversation_history[user_id] = conversation_history[user_id][-10:]
    
    return response

if __name__ == '__main__':
    if not os.path.exists('signatures'):
        os.makedirs('signatures')
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
