import re
import pandas as pd


def preprocess(data):
    # Updated pattern to handle am/pm and extra spaces
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[apAP][mM]\s-\s'

    # Split data into messages and timestamps
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)

    if not messages or not dates:
        raise ValueError("No messages or dates found. Check the regex pattern and file format.")

    # Clean up dates to remove any trailing characters (such as '-')
    dates = [date.strip(' -') for date in dates]

    # Create a DataFrame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date to datetime format
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p', errors='coerce')
    except Exception as e:
        print(f"Error while parsing dates: {e}")

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract user and message from user_message
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', message)
        if entry[1:]:  # If there is a username
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:  # System-generated group notifications
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extract additional date and time information
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Create period (time range) column
    period = []
    for hour in df['hour']:
        if hour == 0:
            period.append(f"12 AM - 1 AM")
        elif hour < 12:
            period.append(f"{hour} AM - {hour + 1} AM")
        elif hour == 12:
            period.append(f"12 PM - 1 PM")
        else:
            period.append(f"{hour - 12} PM - {hour - 11} PM")

    df['period'] = period

    return df
