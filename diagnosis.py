# This script is an enhanced healthcare chatbot that performs disease diagnosis
# based on user-provided symptoms. It uses an ensemble of machine learning
# models for improved accuracy and integrates with the Gemini API to provide
# detailed, grounded explanations of diseases.

import numpy as np
import pandas as pd
import requests
import json
import random
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# IMPORTANT: Set your API key here. The environment will provide it at runtime.
# If you are running this locally, you will need to get an API key from Google AI Studio.
API_KEY = ""
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# --- Embedded Dataset ---
# This data has been extracted directly from your Training.csv and Testing.csv files
# to make the script a single, self-contained file.
training_csv_data = """itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,vomiting,burning_micturition,spotting_ urination,fatigue,weight_gain,anxiety,cold_hands_and_feets,mood_swings,weight_loss,restlessness,lethargy,patches_in_throat,irregular_sugar_level,cough,high_fever,sunken_eyes,breathlessness,sweating,dehydration,indigestion,headache,yellowish_skin,dark_urine,nausea,loss_of_appetite,pain_behind_the_eyes,back_pain,constipation,abdominal_pain,diarrhoea,mild_fever,yellow_urine,yellowing_of_eyes,acute_liver_failure,fluid_overload,swelling_of_stomach,swelled_lymph_nodes,malaise,blurred_and_distorted_vision,phlegm,throat_irritation,redness_of_eyes,sinus_pressure,runny_nose,congestion,chest_pain,weakness_in_limbs,fast_heart_rate,pain_during_bowel_movements,pain_in_anal_region,bloody_stool,irritation_in_anus,neck_pain,dizziness,cramps,bruising,obesity,swollen_legs,swollen_blood_vessels,puffy_face_and_eyes,enlarged_thyroid,brittle_nails,swollen_extremeties,excessive_hunger,extra_marital_contacts,drying_and_tingling_lips,slurred_speech,knee_pain,hip_joint_pain,swelling_joints,painful_walking,small_dents_in_nails,inflammatory_nails,blister,red_sore_around_nose,yellow_crust_ooze,prognosis
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fungal infection
1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fungal infection
1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fungal infection
1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fungal infection
"""

testing_csv_data = """itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,vomiting,burning_micturition,spotting_ urination,fatigue,weight_gain,anxiety,cold_hands_and_feets,mood_swings,weight_loss,restlessness,lethargy,patches_in_throat,irregular_sugar_level,cough,high_fever,sunken_eyes,breathlessness,sweating,dehydration,indigestion,headache,yellowish_skin,dark_urine,nausea,loss_of_appetite,pain_behind_the_eyes,back_pain,constipation,abdominal_pain,diarrhoea,mild_fever,yellow_urine,yellowing_of_eyes,acute_liver_failure,fluid_overload,swelling_of_stomach,swelled_lymph_nodes,malaise,blurred_and_distorted_vision,phlegm,throat_irritation,redness_of_eyes,sinus_pressure,runny_nose,congestion,chest_pain,weakness_in_limbs,fast_heart_rate,pain_during_bowel_movements,pain_in_anal_region,bloody_stool,irritation_in_anus,neck_pain,dizziness,cramps,bruising,obesity,swollen_legs,swollen_blood_vessels,puffy_face_and_eyes,enlarged_thyroid,brittle_nails,swollen_extremeties,excessive_hunger,extra_marital_contacts,drying_and_tingling_lips,slurred_speech,knee_pain,hip_joint_pain,swelling_joints,painful_walking,small_dents_in_nails,inflammatory_nails,blister,red_sore_around_nose,yellow_crust_ooze,prognosis
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fungal infection
"""

def get_ai_disease_info(disease_name):
    """
    Uses the Gemini API with Google Search grounding to get detailed information
    about a specific disease.

    Args:
        disease_name (str): The name of the disease to search for.

    Returns:
        str: A generated text string containing the disease information.
    """
    print(f"I'm searching for information on {disease_name}. Please wait...")
    system_prompt = "Act as a medical informational assistant. Provide a concise, well-structured explanation for the given disease. Include a brief description, common symptoms, causes, and general suggestions for management. Add a clear disclaimer that this is not medical advice and a professional should be consulted."
    user_query = f"Explain the disease: {disease_name}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    try:
        response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No information found.')
        return text
    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching information: {e}"
    except (KeyError, IndexError):
        return "Could not parse the API response. Please try again."

def main_menu():
    """
    Displays the main menu and handles user input.
    """
    while True:
        print("\n--- Health Chatbot Menu ---")
        print("1. Start a new diagnosis")
        print("2. Ask about a disease")
        print("3. Exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            start_diagnosis()
        elif choice == '2':
            ask_about_disease()
        elif choice == '3':
            print("Thank you for using the health chatbot. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def start_diagnosis():
    """
    Runs the main diagnosis loop.
    """
    # Load datasets from embedded strings
    training_dataset = pd.read_csv(io.StringIO(training_csv_data))
    test_dataset = pd.read_csv(io.StringIO(testing_csv_data))

    # Features and labels
    X = training_dataset.iloc[:, :-1].values
    y = training_dataset.iloc[:, -1].values
    
    # Dimensionality reduction for unique disease symptoms
    dimensionality_reduction = training_dataset.groupby(training_dataset['prognosis']).max()
    all_symptoms = list(dimensionality_reduction.columns)

    # Encode target
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train multiple models
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    nb_clf = GaussianNB()

    dt_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)

    # User interaction
    print("\n--- New Diagnosis ---")
    print("Please enter the symptoms you are experiencing. You can enter multiple symptoms separated by commas.")
    print("\nAvailable symptoms are:")
    symptom_chunks = [all_symptoms[i:i + 10] for i in range(0, len(all_symptoms), 10)]
    for chunk in symptom_chunks:
        print(", ".join(chunk))
    
    symptoms_input = input("\nEnter your symptoms (e.g., joint_pain, fatigue, high_fever): ").strip().lower()
    symptoms_list = [symptom.strip() for symptom in symptoms_input.split(',') if symptom.strip()]

    if not symptoms_list:
        print("No symptoms entered. Please try again.")
        return

    # Create input vector
    input_vector = np.zeros(len(all_symptoms))
    symptoms_present = []
    for symptom in symptoms_list:
        try:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1
            symptoms_present.append(symptom)
        except ValueError:
            print(f"Warning: Symptom '{symptom}' not found in the list.")

    if not symptoms_present:
        print("No valid symptoms entered. Exiting diagnosis.")
        return

    sample_input = np.array([input_vector])

    # Make predictions with all models
    dt_disease_code = dt_clf.predict(sample_input)[0]
    rf_disease_code = rf_clf.predict(sample_input)[0]
    nb_disease_code = nb_clf.predict(sample_input)[0]

    dt_disease = labelencoder.inverse_transform([dt_disease_code])[0]
    rf_disease = labelencoder.inverse_transform([rf_disease_code])[0]
    nb_disease = labelencoder.inverse_transform([nb_disease_code])[0]

    # Use combined probabilities for a more robust result
    dt_probs = dt_clf.predict_proba(sample_input)[0]
    rf_probs = rf_clf.predict_proba(sample_input)[0]
    nb_probs = nb_clf.predict_proba(sample_input)[0]

    avg_probs = (rf_probs + nb_probs + dt_probs) / 3
    top_indices = np.argsort(avg_probs)[-3:][::-1]
    top_diseases = labelencoder.inverse_transform(top_indices)

    # Results
    print("\n✅ Based on your symptoms:", ", ".join(symptoms_present))
    print("\n👉 Possible Diseases (Top 3 most likely):")
    for i, disease in enumerate(top_diseases):
        print(f"{i+1}. {disease} (confidence {avg_probs[top_indices[i]]:.2f})")

    # Show predictions from each model
    print("\n📌 Model-specific Predictions:")
    print(f"Decision Tree suggests: {dt_disease}")
    print(f"Random Forest suggests: {rf_disease}")
    print(f"Naive Bayes suggests:   {nb_disease}")

    # Ask for more information
    print("\n--- More Information ---")
    while True:
        choice = input(f"Would you like to know more about '{top_diseases[0]}' (y/n)? ").strip().lower()
        if choice == 'y':
            info = get_ai_disease_info(top_diseases[0])
            print(f"\n{info}\n")
            break
        elif choice == 'n':
            print("You can ask about another disease from the main menu.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def ask_about_disease():
    """
    Allows the user to manually ask for information about a specific disease.
    """
    training_dataset = pd.read_csv(io.StringIO(training_csv_data))
    all_diseases = sorted(training_dataset['prognosis'].unique())

    print("\n--- Ask About a Disease ---")
    print("Here is a list of diseases you can ask about:")
    
    # Print diseases in a few columns for readability
    disease_chunks = [all_diseases[i:i + 5] for i in range(0, len(all_diseases), 5)]
    for chunk in disease_chunks:
        print("\t".join(chunk))

    disease_name = input("\nEnter the name of the disease you want to know about: ").strip()

    if disease_name in all_diseases:
        info = get_ai_disease_info(disease_name)
        print(f"\n{info}\n")
    else:
        print("That disease is not in our list. Please check the spelling.")

if __name__ == "__main__":
    print("Welcome to the AI-Powered Healthcare Chatbot!")
    print("This tool is for informational purposes only and is not a substitute for professional medical advice.")
    print("Always consult with a qualified healthcare provider for any health concerns.")
    main_menu()
