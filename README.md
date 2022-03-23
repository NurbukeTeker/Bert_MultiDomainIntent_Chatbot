# Bert_MultiDomainIntent_Chatbot

 BERT-BASED-UNCASED => ATIS + Banking + ACID + CLINC
codeLink:https://colab.research.google.com/drive/1ojE0zs5JSechs9T0WKRh0qKndNDti9_Z?usp=sharing

![image](https://user-images.githubusercontent.com/37930894/159633735-b761bd49-5bbf-4a40-ad8f-02a99d6bb684.png)





**Ön işleme kısmında 50’den az veriye sahip olan niyetler datasetten atılmıştır.Tüm niyetler domainden bağımsız olarak niyete özgü etiketlenmiştir. 

Tokenizer Info: BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
Max_Len:  30
Model Info: BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = number_of_categories, 
    output_attentions = False,
    output_hidden_states = False)
Model Epoch Sayısı: 4
Optimizer: AdamW
![image](https://user-images.githubusercontent.com/37930894/159633824-4e34883e-99c7-4cae-91fb-42c603a270fa.png)
Accuracy:  0.9097471101913078
F-Score:  0.9114075863381039
Recall:  0.936115229528806
Precision:  0.8950690882265144

Exp6’da yer niyetlerin dağılımının görselleştirilmesi. (50’den az veriye sahip olan veriler drop edilmiştir.)

1. ATIS Dataset total 6 Intent:
![image](https://user-images.githubusercontent.com/37930894/159633962-5ab51cb4-15c9-4981-baf8-ab05ea601028.png)

2. Banking Dataset total 75 Intent:
![image](https://user-images.githubusercontent.com/37930894/159633992-65178950-828d-4e79-ab6a-c66cf7544693.png)

3. CLINC Dataset total 151 Intent:
![image](https://user-images.githubusercontent.com/37930894/159634024-80e8a14b-2099-476f-9e53-237f127530f6.png)

4. ACID Dataset  total 93 Intent:
![image](https://user-images.githubusercontent.com/37930894/159634057-930a8501-2e24-4a5f-a689-8384a15fed63.png)


