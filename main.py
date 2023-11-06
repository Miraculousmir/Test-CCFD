# Security
import hashlib

# Data Analysis
import os

import numpy as np
import pandas as pd

# Frontend Development
import streamlit as st

# Prediction
from sklearn.model_selection import train_test_split
import sklearn.linear_model

# DB Management
import sqlite3


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text

    return False


conn = sqlite3.connect("data.db")
c = conn.cursor()


# DB  Functions
def create_usertable():
    c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)")


def add_userdata(username, password):
    c.execute("INSERT INTO userstable(username,password) VALUES (?,?)", (username, password))
    conn.commit()


def login_user(username, password):
    c.execute("SELECT * FROM userstable WHERE username =? AND password = ?", (username, password))
    data = c.fetchall()
    return data


def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if BEGIN_DATE + '.pkl' <= f <= END_DATE + '.pkl']
    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)

    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    #  Note: -1 are missing values for real world data
    df_final = df_final.replace([-1], 0)

    return df_final


def generate_transaction(start_date="2018-04-01"):
    customer_transaction = []
    customer_id = st.number_input("Customer ID", value=None, step=1)
    terminal_id = st.number_input("Terminal ID", value=None, step=1)
    day = st.number_input("Day", value=0, step=1)
    time_tx = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=None)
    customer_transaction.append([time_tx + day * 86400, day, customer_id, terminal_id, amount])
    customer_transaction = pd.DataFrame(customer_transaction, columns=["TX_TIME_SECONDS", "TX_TIME_DAYS", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT"])
    transactions_df = pd.read_csv("newTransactions.csv")
    customer_transaction["TRANSACTION_ID"] = max(transactions_df["TRANSACTION_ID"]) + 1
    customer_transaction["TX_DATETIME"] = pd.to_datetime(customer_transaction["TX_TIME_SECONDS"], unit="s", origin=start_date)
    customer_transaction = customer_transaction[["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS"]]
    return customer_transaction


def is_weekend(tx_datetime):
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = tx_datetime.weekday()
    # Binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday >= 5

    return int(is_weekend)


def is_night(tx_datetime):
    # Get the hour of the transaction
    tx_hour = tx_datetime.hour
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = tx_hour <= 6

    return int(is_night)


def get_customer_spending_behaviour_features(transaction_df, customer_transaction, windows_size_in_days=[1, 7, 30]):
    # Let us first order transactions chronologically
    new_transaction_df = pd.concat([transaction_df, customer_transaction], axis=0, ignore_index=True)
    new_transaction_df = new_transaction_df.sort_values('TX_DATETIME')

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    new_transaction_df.index = new_transaction_df.TX_DATETIME

    # For each window size
    for window_size in windows_size_in_days:
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW = new_transaction_df['TX_AMOUNT'].rolling(str(window_size) + 'd').sum()
        NB_TX_WINDOW = new_transaction_df['TX_AMOUNT'].rolling(str(window_size) + 'd').count()

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

        # Save feature values
        customer_transaction['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)[len(list(NB_TX_WINDOW)) - 1]
        customer_transaction['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(AVG_AMOUNT_TX_WINDOW)[len(list(AVG_AMOUNT_TX_WINDOW)) - 1]

    # Reindex according to transaction IDs
    new_transaction_df.index = new_transaction_df.TRANSACTION_ID

    # And return the dataframe with the new features
    return customer_transaction


def get_count_risk_rolling_window(terminal_transactions, customer_transaction, delay_period=7, windows_size_in_days=[1, 7, 30], feature="TERMINAL_ID"):
    new_terminal_transactions = pd.concat([terminal_transactions, customer_transaction], axis=0, ignore_index=True)
    new_terminal_transactions = new_terminal_transactions.sort_values('TX_DATETIME')

    new_terminal_transactions.index = new_terminal_transactions.TX_DATETIME

    NB_FRAUD_DELAY = new_terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
    NB_TX_DELAY = new_terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').count()

    for window_size in windows_size_in_days:
        NB_FRAUD_DELAY_WINDOW = new_terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').sum()
        NB_TX_DELAY_WINDOW = new_terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').count()

        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        customer_transaction[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)[len(list(NB_TX_WINDOW)) - 1]
        customer_transaction[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)[len(list(RISK_WINDOW)) - 1]

    new_terminal_transactions.index = new_terminal_transactions.TRANSACTION_ID

    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    customer_transaction.fillna(0, inplace=True)

    return customer_transaction


def main():
    """Simple Login App"""

    st.title("Credit Card Fraud Detection System")
    menu = ["Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login Section")
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:
                st.success("Logged In as {}".format(username))
                task = st.selectbox("Task", ["Show Customer Profiles", "Show Terminal Profiles", "Show Transactions", "Make Prediction", "Blacklist Terminals"])
                if task == "Show Customer Profiles":
                    st.subheader("Customer Profiles")
                    data = pd.read_csv("customerProfiles.csv")
                    customer_id = st.number_input("Customer ID", value=None, step=1)
                    st.table(data[data.CUSTOMER_ID == customer_id])
                elif task == "Show Terminal Profiles":
                    st.subheader("Terminal Profiles")
                    data = pd.read_csv("terminalProfiles.csv")
                    terminal_id = st.number_input("Terminal ID", value=None, step=1)
                    st.table(data[data.TERMINAL_ID == terminal_id])
                elif task == "Show Transactions":
                    st.subheader("Transactions")
                    data = pd.read_csv("newTransactions.csv")
                    customer_id = st.number_input("Customer ID", value=None, step=1)
                    st.table(data[data.CUSTOMER_ID == customer_id])
                elif task == "Make Prediction":
                    st.subheader("Make Prediction")
                    customer_transaction = generate_transaction(start_date="2018-04-01")

                    # create a button to submit input and get prediction
                    submit = st.button("Submit")
                    if submit:
                        customer_transaction['TX_FRAUD_SCENARIO'] = 0
                        customer_transaction['TX_DURING_WEEKEND'] = customer_transaction.TX_DATETIME.apply(is_weekend)
                        customer_transaction['TX_DURING_NIGHT'] = customer_transaction.TX_DATETIME.apply(is_night)
                        transactions_df = read_from_files('tx1', "2018-04-01", "2018-09-30")
                        customer_transaction = get_customer_spending_behaviour_features(transactions_df[transactions_df.CUSTOMER_ID == customer_transaction.at[0, 'CUSTOMER_ID']], customer_transaction)
                        transactions_df = read_from_files('tx2', "2018-04-01", "2018-09-30")
                        customer_transaction = get_count_risk_rolling_window(transactions_df[transactions_df.TERMINAL_ID == customer_transaction.at[0, 'TERMINAL_ID']], customer_transaction, delay_period=7, windows_size_in_days=[1, 7, 30])

                        # loading the dataset to a Pandas DataFrame
                        credit_card_data = pd.read_csv('creditCard.csv')

                        # separating the data for analysis
                        legit = credit_card_data[credit_card_data.TX_FRAUD == 0]
                        fraud = credit_card_data[credit_card_data.TX_FRAUD == 1]

                        # undersample legitimate transactions to balance the classes
                        legit_sample = legit.sample(n=len(fraud), random_state=2)
                        new_dataset = pd.concat([legit_sample, fraud], axis=0)

                        # split data into training and testing sets
                        X = new_dataset[['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW']]
                        Y = new_dataset['TX_FRAUD']
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

                        # train logistic regression model
                        model = sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
                        model.fit(X_train, Y_train)
                        customer_df = pd.read_csv('customerProfiles.csv')
                        customer_id = customer_transaction.loc[0, 'CUSTOMER_ID']
                        terminal_id = customer_transaction.loc[0, 'TERMINAL_ID']
                        blacklist = eval(customer_df.loc[customer_df['CUSTOMER_ID'] == customer_id, 'BLACKLIST'].iloc[0])
                        if terminal_id in blacklist:
                            st.write("Terminal blacklisted, no transaction can be made")
                        else:
                            # get input feature values
                            X_input = customer_transaction[['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW']]
                            # make prediction
                            prediction = model.predict(X_input)
                            # display result
                            if prediction[0] == 0:
                                st.write("Legitimate transaction")
                            else:
                                st.write("Fraudulent transaction")

                            customer_transaction['TX_FRAUD'] = prediction[0]
                            st.table(customer_transaction)
                            data = pd.read_csv("transactions.csv")
                            new_transaction_df = pd.concat([data, customer_transaction], axis=0, join='inner')
                            new_transaction_df.to_csv('newTransactions.csv', index=False)
                elif task == "Blacklist Terminals":
                    st.subheader("Blacklist Terminals")
                    customer_id = st.number_input("Customer ID")
                    terminal_id = st.number_input("Terminal ID")
                    submit = st.button("Submit")
                    customer_df = pd.read_csv("customerProfiles.csv")
                    blacklist_row = eval(customer_df.loc[customer_df['CUSTOMER_ID'] == 0, 'BLACKLIST'].iloc[0])
                    if submit:
                        blacklist_row.extend([int(terminal_id)])
                        customer_df.loc[customer_df['CUSTOMER_ID'] == 0, 'BLACKLIST'] = customer_df.loc[customer_df['CUSTOMER_ID'] == 0, 'BLACKLIST'].apply(lambda x: blacklist_row)
                        customer_df.to_csv('customerProfiles.csv', index=False)
            else:
                st.warning("Incorrect Username/Password")
    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main()
