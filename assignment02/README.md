მოცემულ ნაშრომში ჩატარებულ იქნა სხვადასხვა IP-მისამართების მხრიდან გარკვეულ დროში აქტივობის სტატისტიკური და რეგრესიული ანალიზი სავარაუდო DDoS-შეტევის გამოვლენის მიზნით.
ამ მიზნით გამოყენებულ იქნა პითონის მონაცემთა ანალიზის და, ასევე, ფორმატირების ბიბლიოთეკები (re).
თავდაპირველად მოვლენათა ფაილიდან ამოგდებულ იქნა ყველა ინფორმაცია, გარდა IP-მისამართებისა და აქტივობის დროსა:

1.1. მოვლენათა ფაილის შესაბამისი ფორმატის მქონე შაბლონი, რომელიც გამოვიყენე საჭირო ინფორმაციის ამოსაღებად:

    log_pattern = re.compile(r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>.*?)\]')
   
1.2. ციკლი, რომელიც ზემოთხსენებული შაბლონით მთელ ფაილს პარსავს:

    for line in log_lines:
        match = log_pattern.search(line)
        if match:
            ip_list.append(match.group("ip"))
            timestamp_list.append(match.group("timestamp"))
            
1.3. ახალი მონაცემთა ჩარჩო, რომელიც მზადაა ანალიზისთვის:

    log_df = pd.DataFrame({"IP": ip_list, "Timestamp": timestamp_list})


2.1. შემდეგ, ანალიზის ჩასატრებლად დაჯგუფებულ იქნა IP-მისამართები, მოვლენათა დროები და შერწყმულ იქნა ეს ორი ახალი აგრეგირებული სვეტი ახალ მონაცემთა ჩარჩოდ:

    pps = log_df.groupby('Timestamp').size().rename("PPS")
    usip = log_df.groupby('Timestamp')['IP'].nunique().rename("USIP")
    traffic_df = pd.concat([pps, usip], axis=1).reset_index()

3.1. განხორციელდა უკანასკნელი მოდიფიკაცია: გამოითვალა წრფივი რეგრესიის კოეფიციენტები: დახრა და გადაკვეთა:

    slope, intercept = np.polyfit(X, Y, deg=1)

3.2. გამოითვალა სავარაუდო მნიშვნელობები და ცდმილება რეგრესიის ხაზიდან ზედა კოეფიციენტების გამოყენებით:

    traffic_df['Predicted USIP'] = slope * X + intercept
    traffic_df['Residual'] = Y - traffic_df['Predicted USIP']
    traffic_df['Absolute Residual'] = traffic_df['Residual'].abs()
    
4.1. ვიზუალიზაციამდე გამოითვალა სავარაუდო შეტევის მონაცემები შემდეგი კოდის გამოყენებით:
   
    outlier_rows = traffic_df.sort_values(by='Residual', ascending=False).head(3)
    print(">>> Top 3 Potential DDoS Attack Windows Detected:\n")
        for i, row in outlier_rows.iterrows():
            print(f"Outlier #{i+1}")
            print(row[['Timestamp', 'PPS', 'USIP', 'Predicted USIP', 'Residual']])
            print("-" * 50)

გაირკვა, რომ სავარაუდო შეტევის დრო შეიძლება იყოს 2024-03-22 18:43:28, 2024-03-22 18:43:27 ან 2024-03-22 18:43:39.

უფრო დაწვრილებით:

https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment02/DDoS-attack%20outliers.png

მეტი თვალსაჩინოებისთვის გაეცანით ვიზუალურ მასალას

![DDoS Regression Plot] https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment02/DDoS-attack%20detection.png

მოვლენათა ფაილი

[`nodar_melkonyan_1_server.log`] (https://github.com/Nodar-Melkonyan/ml25spring/blob/main/assignment02/nodar_melkonyan_1_server.log)

