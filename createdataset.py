import pandas as pd
import numpy as np
from sklearn.utils import shuffle
datavariable = pd.read_csv('/content/data/filename.csv')

# datavariable.replace('', np.nan, inplace=True)

datavariable = shuffle(datavariable)

datavariable.head(n=10)

training_data = []

# define new entity label  (LOCATION)
label = 'LOCATION'

sentences_templates = ["I live in {0}", "Friend of mine lives in {0}", "I know the address of that street, it is {0}",
                       "Mr. Absalon Adam lived before in {0}", "I like the resturants in {0}",
                       "Check the map to find the directions to {0}",
                       "My friend Aksel will meet me in {0}", "Me and Adrian has a meeting in {0}",
                       "Taxi driver can take you to {0}",
                       "Stay away of {0}", "His address is {0}", "My cousine lives in {0}", "I like shops in {0}",
                       "Let us drive to {0}",
                       "Do you like this street?", "I walk everyday in that place", "{0}, this is my current address",
                       "{0} is awesome place"
                       ]


# prepare the new training dataset based on the data collected from the street addresses in Kalundborg city in Denmark
def prepare_training_data():
    for index, item in datavariable.iterrows():
        sentence_pholder_idx = np.random.randint(0, len(sentences_templates), size=1)[0]
        sentence_pholder = sentences_templates[sentence_pholder_idx]

        street_address = item['Street Name'] + " " + str(item['road code'])
        if '{0}' in sentence_pholder:
            if item['Additional city name'] != '':
                if item['Additional city name'] != np.nan and str(item['Additional city name']) != 'nan':
                    street_address = street_address + ", " + item['Additional city name']

            street_address = street_address + ", " + item['Municipality']
            start_idx = sentence_pholder.find('{0}')
            new_sentence = sentence_pholder.replace('{0}', street_address)
            end_idx = start_idx + len(street_address)

            training_data.append((new_sentence, {
                'entities': [(start_idx, end_idx, label)]
            }))
        else:
            new_sentence = sentence_pholder

            training_data.append((new_sentence, {
                'entities': []
            }))

    return training_data


dataset = prepare_training_data()
