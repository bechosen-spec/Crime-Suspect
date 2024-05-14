import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model and preprocessing objects
model = joblib.load('crime_model.pkl')
scaler = joblib.load('scaler.pkl')
area_name_encoder = joblib.load('area_name_encoder.pkl')
crime_code_description_encoder = joblib.load('crime_code_description_encoder.pkl')
victim_sex_encoder = joblib.load('victim_sex_encoder.pkl')
victim_descent_encoder = joblib.load('victim_descent_encoder.pkl')
status_code_encoder = joblib.load('status_code_encoder.pkl')
day_of_week_encoder = joblib.load('day_of_week_encoder.pkl')
age_category_encoder = joblib.load('age_category_encoder.pkl')

# Define the prediction function with preprocessing
def predict_suspect(features):
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])

    # Preprocess categorical features
    features_df['Area Name'] = area_name_encoder.transform(features_df['Area Name'])
    features_df['Crime Code Description'] = crime_code_description_encoder.transform(features_df['Crime Code Description'])
    features_df['Victim Sex'] = victim_sex_encoder.transform(features_df['Victim Sex'])
    features_df['Victim Descent'] = victim_descent_encoder.transform(features_df['Victim Descent'])
    features_df['Status Code'] = status_code_encoder.transform(features_df['Status Code'])
    features_df['Day of Week Reported'] = day_of_week_encoder.transform(features_df['Day of Week Reported'])
    features_df['Day of Week Occurred'] = day_of_week_encoder.transform(features_df['Day of Week Occurred'])
    features_df['Age Category'] = age_category_encoder.transform(features_df['Age Category'])

    # Scale numerical features
    numerical_features = ['DR Number', 'Time Occurred', 'Reporting District', 'Victim Age',
                          'Premise Code', 'Weapon Used Code', 'Crime Code 1', 'Day Reported',
                          'Month Occurred', 'Day Occurred']
    features_df[numerical_features] = scaler.transform(features_df[numerical_features])

    # Make a prediction
    prediction = model.predict(features_df)
    return 'Suspect' if prediction[0] else 'Not a Suspect'

# Streamlit app
st.title('Crime Suspect Prediction')

# Input fields for features
st.header('Enter the Details:')
dr_number = st.number_input('DR Number', value=123456)
time_occurred = st.number_input('Time Occurred (HHMM)', value=2300)
area_name = st.selectbox('Area Name', ['Olympic', 'Southeast', 'Northeast', 'Foothill', 'Mission', 'Newton',
                                       'West Valley', '77Th Street', 'Pacific', 'N Hollywood', 'Topanga',
                                       'Devonshire', 'Rampart', 'Central', 'Southwest', 'Hollenbeck', 'Hollywood',
                                       'Harbor', 'West La', 'Wilshire', 'Van Nuys'])
reporting_district = st.number_input('Reporting District', value=101)
crime_code_description = st.selectbox('Crime Code Description', ['Vehicle - Stolen', 'Burglary From Vehicle', 'Indecent Exposure',
                                                                 'Other Assault', 'Other Miscellaneous Crime',
                                                                 'Embezzlement, Grand Theft ($950.01 & Over)', 'Disturbing The Peace',
                                                                 'Burglary', 'Child Annoying (17Yrs & Under)', 'Theft Plain - Petty ($950 & Under)',
                                                                 'Vandalism - Misdeameanor ($399 Or Under)', 'Cruelty To Animals',
                                                                 'Theft Of Identity', 'Arson', 'Theft, Person', 'Burglary, Attempted',
                                                                 'Trespassing', 'Theft From Motor Vehicle - Petty ($950 & Under)',
                                                                 'Violation Of Court Order', 'Attempted Robbery', 'Letters, Lewd  -  Telephone Calls, Lewd',
                                                                 'Robbery', 'Crm Agnst Chld (13 Or Under) (14-15 & Susp 10 Yrs Older)', 'Rape, Forcible',
                                                                 'Purse Snatching', 'Bike - Stolen', 'Brandish Weapon', 'Counterfeit',
                                                                 'Lewd Conduct', 'Violation Of Restraining Order', 'Intimate Partner - Simple Assault',
                                                                 'Stalking', 'Pimping', 'Extortion', 'Pickpocket', 'Reckless Driving', 'Vehicle - Attempt Stolen',
                                                                 'Theft From Person - Attempt', 'Battery - Simple Assault', 'Battery Police (Simple)',
                                                                 'Intimate Partner - Aggravated Assault', 'Theft From Motor Vehicle - Grand ($400 And Over)',
                                                                 'Child Neglect (See 300 W.I.C.)', 'Vandalism - Felony ($400 & Over, All Church Vandalisms)',
                                                                 'Criminal Threats - No Weapon Displayed', 'Theft-Grand ($950.01 & Over)Excpt,Guns,Fowl,Livestk,Prod',
                                                                 'Resisting Arrest', 'Document Forgery / Stolen Felony', 'Child Abuse (Physical) - Simple Assault',
                                                                 'Driving Without Owner Consent (Dwoc)', 'Shoplifting - Petty Theft ($950 & Under)',
                                                                 'Assault With Deadly Weapon, Aggravated Assault', 'False Imprisonment', 'Kidnapping - Grand Attempt',
                                                                 'Criminal Homicide', 'Defrauding Innkeeper/Theft Of Services, $400 & Under', 'Discharge Firearms/Shots Fired',
                                                                 'Credit Cards, Fraud Use ($950.01 & Over)', 'Kidnapping', 'Bomb Scare', 'Dishonest Employee - Grand Theft',
                                                                 'Unauthorized Computer Access', 'Burglary From Vehicle, Attempted', 'Threatening Phone Calls/Letters',
                                                                 'Battery With Sexual Contact', 'Theft From Motor Vehicle - Attempt', 'Shoplifting-Grand Theft ($950.01 & Over)',
                                                                 'Contributing', 'Embezzlement, Petty Theft ($950 & Under)', 'Bunco, Petty Theft', 'False Police Report',
                                                                 'Bunco, Attempt', 'Child Abuse (Physical) - Aggravated Assault', 'Sex,Unlawful(Inc Mutual Consent, Penetration W/ Frgn Obj',
                                                                 'Bunco, Grand Theft', 'Assault With Deadly Weapon On Police Officer', 'Throwing Object At Moving Vehicle',
                                                                 'Child Stealing', 'Child Abandonment', 'Theft Plain - Attempt', 'Weapons Possession/Bombing', 'Illegal Dumping',
                                                                 'Rape, Attempted', 'Purse Snatching - Attempt', 'Peeping Tom', 'Oral Copulation', 'Shots Fired At Inhabited Dwelling',
                                                                 'Prowler', 'Sodomy/Sexual Contact B/W Penis Of One Pers To Anus Oth', 'Violation Of Temporary Restraining Order',
                                                                 'Grand Theft / Insurance Fraud', 'Contempt Of Court', 'Battery On A Firefighter', 'Boat - Stolen',
                                                                 'Theft, Coin Machine - Grand ($950.01 & Over)', 'Shots Fired At Moving Vehicle, Train Or Aircraft',
                                                                 'Credit Cards, Fraud Use ($950 & Under', 'Pandering', 'Sexual Penetration W/Foreign Object', 'Dishonest Employee - Petty Theft',
                                                                 'Conspiracy', 'Defrauding Innkeeper/Theft Of Services, Over $400', 'Petty Theft - Auto Repair', 'Disrupt School',
                                                                 'Till Tap - Petty ($950 & Under)', 'Document Worthless ($200.01 & Over)', 'Bribery', 'Inciting A Riot',
                                                                 'Failure To Yield', 'Shoplifting - Attempt', 'Beastiality, Crime Against Nature Sexual Asslt With Anim',
                                                                 'Theft, Coin Machine - Petty ($950 & Under)', 'Telephone Property - Damage', 'Drunk Roll', 'Manslaughter, Negligent',
                                                                 'Lynching', 'Till Tap - Attempt', 'Till Tap - Grand Theft ($950.01 & Over)', 'Pickpocket, Attempt',
                                                                 'Grand Theft / Auto Repair', 'Bigamy', 'Human Trafficking - Commercial Sex Acts', 'Drunk Roll - Attempt',
                                                                 'Drugs, To A Minor', 'Lynching - Attempted', 'Theft, Coin Machine - Attempt', 'Document Worthless ($200 & Under)',
                                                                 'Lewd/Lascivious Acts With Child', 'Failure To Disperse', 'Bike - Attempted Stolen',
                                                                 'Replica Firearms(Sale,Display,Manufacture Or Distribute)', 'Human Trafficking - Involuntary Servitude',
                                                                 'Child Pornography', 'Abortion/Illegal', 'Dishonest Employee Attempted Theft', 'Incest (Sexual Acts Between Blood Relatives)',
                                                                 'Blocking Door Induction Center', 'Train Wrecking', 'Firearms Restraining Order (Firearms Ro)',
                                                                 'Firearms Temporary Restraining Order (Temp Firearms Ro)'])
victim_age = st.number_input('Victim Age', value=30)
victim_sex = st.selectbox('Victim Sex', ['Unknown', 'F', 'M', 'X', 'H', '-', 'N'])
victim_descent = st.selectbox('Victim Descent', ['Unknown', 'W', 'H', 'X', 'O', 'B', 'A', 'K', 'P', 'C', 'G', 'F', 'J', 'I', 'V', 'U', 'S', 'Z', 'D', 'L', '-'])
premise_code = st.number_input('Premise Code', value=1.0)
premise_description = st.text_input('Premise Description', value='STREET')
weapon_used_code = st.number_input('Weapon Used Code', value=1)
status_code = st.selectbox('Status Code', ['Invest Cont', 'Adult Arrest', 'Adult Other', 'Juv Other', 'Juv Arrest', 'Unk'])
crime_code_1 = st.number_input('Crime Code 1', value=200.0)
address = st.text_input('Address', value='Unknown')
day_reported = st.number_input('Day Reported', value=25.0)
day_of_week_reported = st.selectbox('Day of Week Reported', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
month_occurred = st.number_input('Month Occurred', value=1.0)
day_occurred = st.number_input('Day Occurred', value=22.0)
day_of_week_occurred = st.selectbox('Day of Week Occurred', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
age_category = st.selectbox('Age Category', ['Child', 'Teen', 'Adult', 'Senior'])

# Collect the features into a dictionary
features = {
    'DR Number': dr_number,
    'Time Occurred': time_occurred,
    'Area Name': area_name,
    'Reporting District': reporting_district,
    'Crime Code Description': crime_code_description,
    'Victim Age': victim_age,
    'Victim Sex': victim_sex,
    'Victim Descent': victim_descent,
    'Premise Code': premise_code,
    'Premise Description': premise_description,
    'Weapon Used Code': weapon_used_code,
    'Status Code': status_code,
    'Crime Code 1': crime_code_1,
    'Address': address,
    'Day Reported': day_reported,
    'Day of Week Reported': day_of_week_reported,
    'Month Occurred': month_occurred,
    'Day Occurred': day_occurred,
    'Day of Week Occurred': day_of_week_occurred,
    'Age Category': age_category
}

# Predict button
if st.button('Predict'):
    result = predict_suspect(features)
    st.write(f'Prediction: {result}')
