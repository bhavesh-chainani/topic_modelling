import wikipedia
import pandas as pd

#selective wikipedia article topics
articles = ['George_Clooney', 'Shah_Rukh_Khan', 'Leonardo_DiCaprio', 'Will_Smith',
            'Kamal_Haasan', 'Tom_Cruise', 'Dwayne_Johnson', 'Brad_Pitt',
            'Johnny_Depp', 'Morgan_Freeman', 'Ed_Sheeran', 'A._R._Rahman',
            'Bruno_Mars', 'Taylor_Swift', 'Eminem', 'Shakira', 'Ellie_Goulding',
            'Michael_Jackson', 'Selena_Gomez', 'Lionel_Messi',
            'Cristiano_Ronaldo', 'Rafael_Nadal', 'Usain_Bolt',
            'Stephen_Curry', 'Roger_Federer', 'Virat_Kohli', 'Serena_Williams',
            'James_Patterson', 'Stephen_King', 'J._K._Rowling',
            'Dan_Brown', 'Agatha_Christie', 'Ken_Follett', 'Neil_Gaiman', 
            'John_Grisham', 'Nora_Roberts', 'Arundhati_Roy', 'Mark_Zuckerberg',
            'Jeff_Bezos', 'Bill_Gates', 'Larry_Page', 'Tim_Cook', 
            'Elon_Musk', 'Warren_Buffett', 'Xi_Jinping', 'Vladimir_Putin', 
            'Donald_Trump', 'Barack_Obama', 'David_Cameron', 'Hillary_Clinton',
            'Bill_Clinton']

print("Obtaining English data now")
wikipedia.set_lang("en")
content = []
person = []

for i in articles:
#     print("Examining " + str(i))
    person.append(wikipedia.page(i, auto_suggest=False).title)
    content.append(wikipedia.page(i, auto_suggest=False).content)
    
df_eng = pd.DataFrame(list(zip(person,content)), columns=['person', 'content'])
print("English dataframe has been created")

print("Obtaining Russian data now")
wikipedia.set_lang("ru")
content = []
person = []

for i in articles:
#     print("Examining " + str(i))
    person.append(wikipedia.page(i, auto_suggest=False).title)
    content.append(wikipedia.page(i, auto_suggest=False).content)
    
df_ru = pd.DataFrame(list(zip(person,content)), columns=['person', 'content'])
print("Russian dataframe has been created")

print("Obtaining Italian data now")
wikipedia.set_lang("it")
content = []
person = []

for i in articles:
#     print("Examining " + str(i))
    person.append(wikipedia.page(i, auto_suggest=False).title)
    content.append(wikipedia.page(i, auto_suggest=False).content)
    
df_it = pd.DataFrame(list(zip(person,content)), columns=['person', 'content'])
print("Italian dataframe has been created")

## combining data into data_comb
df_comb = pd.concat([df_eng, df_it, df_ru], axis=0).reset_index(drop=True)

df_comb.to_excel("./files/input/combined_data.xlsx",index=False)
print("Combined dataframe has been saved in ./files/input folder.")
