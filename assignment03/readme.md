# მოცემული ანგარიში წარმოადგენს ლოგისტიკური რეგრესიის გამოყენებით დასწავლილი მანქანის მიმოხილვას

![პროგრამის კოდი] https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment03/Logistic%20Regression%20for%20Credit%20Data

ამ ამოცანისთვის გამოყენებულ იქნა 'German Credit Risk' მონაცემთა ნაკრები. 

![დამუშავებული მონაცემთა ნაკრები] https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment03/german_credit_with_headers.csv

მონაცემებთან მუშაობისთვის გამოყენებულ იქნა Google Colab-ი.

## ნაბიჯი 1
ბიბლიოთეკების გადმოწერა

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

## ნაბიჯი 2
.data გაფართოების მქონე ფაილის ატვირთვა Google Colab-ზე

    from google.colab import files
    uploaded = files.upload

ოწიგინალური წყაროდან გადმოწერისას აღმოჩნდა, რომ ფაილი ხელმისაწვდომი იყო მხოლოდ .data გაფართოებაში.

![ორიგინალური ფაილი] https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment03/german.data

ამიტომ, საჭირო გახდა მისი დამუშავება .csv ფორმატში გადაყვანის მიზნით:

    file_path = '/content/german.data'
    df = pd.read_csv(file_path, delimiter=' ', header=None)
    df.to_csv('german_credit_20_columns.csv', index=False)

## ნაბიჯი 3 
მონაცემების მიმოხილვა და მონაცემთა ტიპების და რაოდენობის შესახებ ინფორმაციის ამოკითხვა

    df.head()
    df.info()


ასევე, ორიგინალურ ფაილში მონაცემები იყო კოდირებული და ცვეტებს არ ახლდათ სახელეწოდებები.

ეს საკითხი მოგვარდა ორიგინალურ წყაროში თანდართული .doc ფაილის მეოხებით.

![განმარტებითი .doc ფაილი] https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment03/german.doc

## ნაბიჯი 4

    columns = [
        'Status_existing_account', 'Duration', 'Credit_history', 'Purpose',
        'Credit_amount', 'Savings_account', 'Employment_since', 'Installment_rate',
        'Personal_status_sex', 'Other_debtors', 'Present_residence_since',
        'Property', 'Age', 'Other_installment_plans', 'Housing',
        'Number_existing_credits', 'Job', 'Number_people_liable', 'Telephone',
        'Foreign_worker', 'Credit_risk'  # Target (1=Good, 2=Bad)
    ]
    
    df.columns = columns

## ნაბიჯი 5
მეტი თვალსაჩინოებისთვის შეიქმნა .csv ფაილის, რომელიც თანდართულია ამ ფაილის თავში

    df.to_csv('german_credit_with_headers.csv', index=False)

## ნაბიჯი 6
ნულვანი მონაცემების შემოწმება

    df.isnull().sum()

ასეთი არ გამოვლინდა

## ნაბიჯი 7
სამიზნე ცვლადის ბინარულ ფორმატში გადაყვანა მოდელისთვის

    df['Credit_risk'] = df['Credit_risk'].map({1: 1, 2: 0})

## ნაბიჯი 8
კატეგორიული და რიცხვითი ცვლადების გამოყოფა შემდეგი დამუშავებისთვის

    categorical_columns = [
        'Status_existing_account', 'Credit_history', 'Purpose', 'Savings_account', 'Employment_since',
        'Personal_status_sex', 'Other_debtors', 'Property', 'Other_installment_plans', 'Housing', 'Job', 'Telephone', 'Foreign_worker'
    ]
    numerical_columns = list(set(df.columns) - set(categorical_columns) - {'Credit_risk'})


## ნაბიჯი 9
დამოკიდებული (y) და დამოუკიდებელი (X) ცვლადების განსაზღვრა

    X = df.drop('Credit_risk', axis=1)
    y = df['Credit_risk']

## ნაბიჯი 10
კატეგორიული ცვლადების გადაყვანა რიცხვით ფორმატში და რიცხვითი ცვლადების სტანდარტიზაცია

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_columns),
            ('num', StandardScaler(), numerical_columns)
        ]
    )
    X_processed = preprocessor.fit_transform(X)

ზოგიერთი ცვლადის (Credit_amount) მნიშვნელობები ძალიან განსხვავდებოდა სხვებისგან. ამიტომაც, სტანდარტიზაციაც არის გამოყენებული.

აღმოჩნდა, რომ სტანდარტიზებული ცვლადების შემთხვევაში, შედეგები ოდნავ უკეთესი იყო.

## ნაბიჯი 11
hot-enoding-ის მიერ სვეტებისთვის მინიჭებული სახელების ამოკითხვა და თითეული კატეგორიული სვეტისთვის ცალკე სვეტების შექმნა hot-encoding-ის პრინციპების მიხედვით

    encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
    all_feature_names = list(encoded_feature_names) + numerical_columns


მატრიცის ხელახლა მონ. ჩარჩოში გადაყვანა

    X_df = pd.DataFrame(X_processed, columns=all_feature_names)

## ნაბიჯი 12
დამოკიდებული და დამოუკიდებელი ცვლადების გახლეჩა საწვრთნ (70%) და საცდელ (30%) ნაწილებად

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

## ნაბიჯი 13
მოდელის გაწვრთნა

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

## ნაბიჯი 14
შეფასება

    y_pred = model.predict(X_test)
    print("შეფასება:\n", classification_report(y_test, y_pred))


## შედეგების აღწერა


მოდელმა ამოიცნო დადებითი შემთხვევების 90% და ჯამში სწორად განსაზღვრა დადებითი შემთხვევების 80%.

თუმცა, შეფასებისას საინტერესო ტენდენცია გამოვლინდა. მოდელის აკურატულობამ 77% შეადგინა, რაც ზემოთხსენებული შედეგების გათვალისწინებით უცნაურია.

აღმოჩნდა, რომ მოდელი არც ისე კარგად უმკლავდება უარყოფით შემთხვევბს: იცნობს მათ მხოლოდ 68%-ს და სწორად განსაზღვრავს მხოლოდ 47%-ს.

ეს, დიდი ალბათობით, მონაცემთა სიმწირეს უკავშირდება: უაროფითი შემთხვევების საცდელად გამოყენებულ იქნა 91 სესხი, ხოლო დადებითების — 209.

![შეფასების ანგარიში] https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment03/Evaluation.png
