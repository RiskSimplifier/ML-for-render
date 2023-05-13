
import streamlit_authenticator as stauth

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import base64
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import numpy as np



###check this video https://www.youtube.com/watch?v=eCbH2nPL9sU&t=16s regarding how not to expose your database key in heroku

def main():
   
     
    
    st.title("Transaction Monitoring Simplified ")
    st.write()
    st.sidebar.markdown('====================================')
    st.sidebar.title("Working Area")
    st.markdown("Just Upload, Click Button and see Magic Happen !!!")


    # ## fileuploader

    st.sidebar.markdown('---')
    st.sidebar.subheader("Let's Train our Algorithm")

    uploaded_file = st.sidebar.file_uploader("Upload Training  Data CSV file", type=["csv"],key='file_uploader_1')
    save_button = st.sidebar.button('save file')
    if save_button:
        save_file = uploaded_file
        with open(save_file.name, "wb") as f:
                f.write(save_file.getbuffer()) 
        st.write("File saved successfully.")
    else:
        st.sidebar.warning('Please select the file you want to upload')

        
    def load_data():
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df)
            column_headers= df.columns
            labelencoder=LabelEncoder()
            if 'Risk Rating' in column_headers:
                Risk_enc= labelencoder.fit_transform(df['Risk Rating'])     
                df.drop(columns=['Risk Rating'])
                df['Risk Rating'] = Risk_enc
            STR_target = df['STR']
            df= df.drop(columns=['STR'])
            scaler = StandardScaler()
            df = pd.DataFrame(scaler.fit_transform(df),columns =df.columns)
            df['STR']= STR_target 
            data = df      
            return data

        except:
            print("Please Upload file to start the program!!!")

    
    def predict_data():
            try:
                df = pd.read_csv(predict_file)
                st.write(df)
                column_headers= df.columns
                labelencoder=LabelEncoder()
                if 'Risk Rating' in column_headers:
                    Risk_enc= labelencoder.fit_transform(df['Risk Rating'])     
                    df.drop(columns=['Risk Rating'])
                    df['Risk Rating'] = Risk_enc
                scaler = StandardScaler()
                df= pd.DataFrame(scaler.fit_transform(df),columns =df.columns)
                data =df.fillna(df.mean())
                
                return data

            except:
                print("Please Upload file to start the program!!!")   


    
    df1 = load_data() 
    
    class_names = ['STR', 'NotSTR']
    
    # if st.sidebar.checkbox("Show training raw data", False):
    #     st.subheader("This is the training data")
    #     st.write(df1)
        

    
        



    
    def split(df1):
        try:
            X = df1.drop(['STR'],axis=1)
            y = df1['STR']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            return X_train, X_test, y_train, y_test
        except:
            print("Please Upload file to start the program!!!")


    try:
        X_train, X_test, y_train, y_test = split(df1)
    except:
        print("Please Upload file to start the program!!!")

    def plot_graph(visual_list):
        
        if 'Features Ranking' in visual_list:
            st.subheader("Random Forest Variable Importance Calculator")
                # Split the dataset into features and target
            X = df1.drop(columns=["STR"])
            y = df1["STR"]

            rf = ran_model()

            # Get the feature importances from the trained model and store them in a dictionary
            importance_dict = {}
            for feature, importance in zip(X.columns, rf.feature_importances_):
                importance_dict[feature] = importance

            # Sort the dictionary by values in descending order
            sorted_importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

            # Create a horizontal bar chart to show the sorted feature importance values for each variable
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.barh(range(len(sorted_importance_dict)), list(sorted_importance_dict.values()), align='center')
            ax.set_yticks(range(len(sorted_importance_dict)))
            ax.set_yticklabels(list(sorted_importance_dict.keys()))
            ax.set_xlabel("Feature Importance")
            ax.set_ylabel("Variable")
            st.pyplot(fig)



    
        if 'Correlation Matrix' in visual_list:
            st.subheader('Correlationg Matrix')
            
            # Compute the correlation matrix
            corr = df1.corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.pcolor(corr.values, cmap='coolwarm')
            fig.colorbar(im)
            ax.set_xticks(np.arange(len(corr.columns))+0.5, minor=False)
            ax.set_yticks(np.arange(len(corr.index))+0.5, minor=False)
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.index)
            ax.set_title("Correlation Matrix (Matplotlib)")
            st.pyplot(fig)

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        
        if 'Classfication Report' in metrics_list:
            st.subheader('Classification Report')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            st.text(report)


    
    st.sidebar.markdown('---')
    st.sidebar.subheader("Random Forest Algorithm")

    def ran_model():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, criterion = 'entropy', n_jobs=-1)
        model.fit(X_train, y_train)
        return model

    
    
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

    

    metrics = st.sidebar.multiselect("Measure performance of your Model.", ('Confusion Matrix', 'Classfication Report'))

    if st.sidebar.button("Classify", key='classify'):
        st.subheader("Random Forest Results")
        model = ran_model()
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2) )
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2) )
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2) )
        plot_metrics(metrics)

        
    st.sidebar.markdown('---')
    visual = st.sidebar.multiselect("What visualization to plot?", ('Features Ranking', 'Correlation Matrix'))
    plot_graph(visual)  
        
    st.sidebar.markdown('---')
    predict_file = st.sidebar.file_uploader("Upload CSV file for Prediction", type=["csv"],key='file_uploader_2')

    # predict_button = st.sidebar.button('Upload')
    # if predict_button:
    #         predict_file = predict_file
    #         with open(predict_file.name, "wb") as f:
    #                 f.write(predict_file.getbuffer()) 
    #         st.write("File Uploaded successfully.")
                    
    # else:
    #         st.sidebar.warning('Please select the file you want to upload')       

    

    df2 = predict_data()   
    # if st.sidebar.checkbox("Show predict raw data", False):
    #         st.subheader("This is the Predicted data")
    #         st.write(df2)
        

    if st.sidebar.button("Predict", key='predict'):
            # randomforest_classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
            # randomforest_classifier.fit(X_train, y_train)
            model = ran_model()
            y_predict = model.predict(df2)
            data = pd.DataFrame({
                        "STR": [ ]
                            })
            df_X= pd.DataFrame(y_predict)
                    

        
                    

            # # Save the DataFrame as a CSV file
            csv = df_X.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()

            # # Create a download link for the CSV file
            href = f'<a href="data:file/csv;base64,{b64}" download="PREDICT.csv">Download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)                   

        
       
               
if __name__ == '__main__':
    main()
