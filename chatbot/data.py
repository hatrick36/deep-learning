import json
data = {'data': [
    {'tag': 'greeting',
     'patterns': ["hi", "hello", "hey", "helloo", "hellooo",
                  "good day", "good afternoon", "good evening", "greetings", "greeting", "good to see you",
                  "its good seeing you", "g’day", "howdy"],
     'response': ["yo", "hello", "hi", "whats up", "howdy", "howdy hay", "whats good", "hey", "whats poppin"],
     'context_set': ''},
    {'tag': 'morning_greeting',
     'patterns': ["g morining", "gmorning", "good morning", "morning", "rise and shine"],
     'response': ["hi", "hello", "hey", "helloo", "hellooo", "good morning", "morning",
                  "hello how did you sleep"],
     'context_set': ''},
    {'tag': 'afternoon_greeting',
     'patterns': ["good afternoon", "good evening", "afternoon"],
     'response': ["yo", "hello", "hi", "whats up", "howdy", "howdy hay", "whats good", "hey",
                  "good afternoon", "good evening", "afternoon"],
     'context_set': ''},
    {'tag': 'open_ended_greeting',
     'patterns': ["how are you", "how're you", "how are you doing", "how ya doin'",
                  "how ya doin", "how is everything", "how is everything going", "how's everything going",
                  "how is you", "how's you", "how are things", "how're things", "how is it going", "how's it going",
                  "how's it goin'", "how's it goin", "how's life been treating you",
                  "how have you been", "how've you been", "what is up", "what's up", "what is cracking",
                  "what's cracking", "what is good", "what's good", "what is happening", "what's happening",
                  "what is new", "what's new", "what is neww"],
     'response': ["good", "fine", "not bad", "same old same old", "still a computer and you", "pretty good",
                  "good how are you",
                  "kill all humans just kidding im fine how are you", "I'm fine thank you for asking", "nothing much"],
     'context_set': ''},
    {'tag': 'farewells',
     'patterns': ['bye', 'bye bye', 'see ya', 'farewell', 'see you', 'good bye', 'nice talking too you', 'so long',
                  'have a good one', 'peace', 'see you later', 'talk to you later', 'catch you later',
                  'later', 'so long', 'i will catch you later', 'adios',
                  'thats all', 'thats it', 'no more', 'that is it', 'that is all', 'catch you later'],
     'response': ['bye', 'bye bye', 'see ya', 'farewell', 'see you', 'good bye', 'nice talking too you', 'so long',
                  'have a good one', 'peace', 'see you later', 'talk to you later', 'catch you later',
                  'later', 'so long', 'i will catch you later', 'adios'],
     'context_set': ''},
    {'tag': 'night_time',
     'patterns': ['good night', 'night', 'sleep tight', 'sleep well', 'im going to bed', 'im going to bed', 'night night',
                 'nighty night', 'time to go to bed', 'bed time', 'time for bed', 'bed time'],
     'response': ['good night', 'night', 'sleep tight', 'sleep well'],
     'context_set': ''},
    {'tag': 'sassy',
     'patterns': ["how is life been treating you"],
     'response': ["you realize you are speaking to a machine yes", "I'm a machine"],
     'context_set': ''},
    {'tag': 'name',
     'patterns': ["what's your name", "what do they call you", "name please", "what is your name", "do you have a name"
                                                                                                   "what do your friends call you",
                  "who are you", "and you are"],
     'response': ["my name is gene", "I'm gene", "gene", "gene is the name", "its gene", "I am gene",
                  "friends call me gene", "my friends call me gene", "genes the name", "gene and you"],
     'context_set': ''},
    {'tag': 'identity',
     'patterns': ['what are you', 'are you a', 'gene what are you', 'do you know what you are'],
     "response": ["I'm an AI",
                  "I am an artificailly intelligent chatbot that uses natural language processes with deep \n"
                  "nueral networks to build and recognize patterns in human speech so I can communicate \n"
                  "with humans using their natural tongue"],
     'context_set': ''},
    {'tag': 'favor',
     'patterns': ['can you do me a favor', 'i need a favor', 'help me', 'can you', 'could you', 'would you', 'do me favor',
                  'would you do me a favor', 'i need help', 'please help me', 'could you do me a favor'],
     'response': ['sure', 'what do you need', 'sure what is it', 'no problem', 'of course', 'ya', 'sure thing'],
     'context_set': ''},
    {'tag': 'joke',
     'patterns': ['tell me a joke', 'joke', 'say something funny', 'got any jokes', 'be funny', 'make me laugh',
                  'i need a laugh', 'i wanna hear somthing funny', 'tell a joke'],
     'response': ['i just went to an emotional wedding even the cake was in tiers', 'whens the best time to go to the dentist toothhurtie',
                 'what do you call a dangerous sun shower a rain of terror',
                 'why do seagulls fly over the sea because if they flew over the bay theyve bagels',
                 'what do you call a farm that makes bad jokes corny',
                 'why do fish live in salt water because pepper makes them sneeze',
                 'what streets to ghosts haunt dead ends',
                 'what do you tell actors to break a leg because every play has a cast',
                 'what kind of dogs love car racing lap dogs',
                 'what did winnie the pooh say to his agent show me the honey',
                 'what do you call birds who stick together velcrows',
                 'today i gave my dead batteries away they were free of charge',
                 'the best funny jokes to tell at parties', 'shutterstock',
                 'what do you call it when one cow spies on another a steak out',
                 'what happens when a frogs car breaks down it gets toad',
                 'i went on a onceinalifetime vacation never again',
                 'whats the best thing about switzerland i dont know but its flag is a big plus',
                 'my favorite word is drool it just rolls off the tongue',
                 'why is peter pan always flying he neverlands',
                 'i just wrote a book on reverse psychology do not read it',
                 'what does a zombie vegetarian eat graaaaaaaains',
                 'my new thesaurus is terrible not only that but its also terrible',
                 'why didnt the astronaut come home to his wife he needed his space',
                 'i got fired from my job at the bank today an old lady came in and asked me to check her balance so i pushed her over',
                 'i wasnt going to visit my family this december but my mom promised to make me eggs benedict so im '
                 'going home for the hollandaise', 'what did the blanket say as it fell off the bed oh sheet',
                 'i like to spend every day as if its my last staying in bed and calling for a nurse to bring me more '
                 'pudding', 'why do cowmilking stools only have three legs cause the cows got the udder',
                 'how did darth vader know what luke got him for christmas he felt his presents', 'whats the last '
                                                                                                  'thing that goes '
                                                                                                  'through a bugs '
                                                                                                  'mind when it hits '
                                                                                                  'a windshield its '
                                                                                                  'butt',
                 'what did the janitor say when he jumped out of the closet supplies', 'imagine if americans switched '
                                                                                       'from pounds to kilograms '
                                                                                       'overnight there would be mass '
                                                                                       'confusion',
                 'its inappropriate to make a dad joke if you are not a dad its a faux pa', 'what did batman say to '
                                                                                            'robin before they got in '
                                                                                            'the car robin get in the '
                                                                                            'car', 'i have an '
                                                                                                   'addiction to '
                                                                                                   'cheddar cheese '
                                                                                                   'but its only '
                                                                                                   'mild',
                 'why shouldnt you write with a broken pencil because its pointless', 'why did the scarecrow win an '
                                                                                      'award he was outstanding in '
                                                                                      'his field', 'what did the '
                                                                                                   'buffalo say when '
                                                                                                   'his son left '
                                                                                                   'bison',
                 'the best stupid jokes people cant help but laugh at', 'shutterstock', 'i was sitting in traffic the '
                                                                                        'other day\xa0 probably why i '
                                                                                        'got run over', 'sometimes i '
                                                                                                        'tuck my '
                                                                                                        'knees into '
                                                                                                        'my chest and '
                                                                                                        'lean forward '
                                                                                                        'thats just '
                                                                                                        'how i roll',
                 'whats red and shaped like a bucketa blue bucket painted red', 'what dont ants get sick they have '
                                                                                'antybodies', 'what do you call a '
                                                                                              'fish with no eye '
                                                                                              'fssshh', 'why do you '
                                                                                                        'smear peanut '
                                                                                                        'butter on '
                                                                                                        'the road to '
                                                                                                        'go with the '
                                                                                                        'traffic '
                                                                                                        'jam',
                 'when is your door not actually a door when its actually ajar', 'my grandfather has the heart of a '
                                                                                 'lion and a lifetime ban from the '
                                                                                 'national zoo', 'whats green and has '
                                                                                                 'wheels grass i lied '
                                                                                                 'about the wheels',
                 'whats green fuzzy and would hurt if it fell on you out of a tree a pool table', 'a communist joke '
                                                                                                  'isnt funny unless '
                                                                                                  'everyone gets it',
                 'what did one dish say to the other dinner is on me', 'what does a house wear address',
                 'why cant you hear a pterodactyl go to the bathroom because the pee is silent', 'cosmetic surgery '
                                                                                                 'used to be such a '
                                                                                                 'taboo subject now '
                                                                                                 'you can talk about '
                                                                                                 'botox and nobody '
                                                                                                 'raises an eyebrow',
                 'what do you call someone who immigrated to swedenartificial swedener', 'have you heard the one '
                                                                                         'about the corduroy pillow '
                                                                                         'its making headlines',
                 'what do you call a man with no arms and no legs in a pool bob', 'what do you call a man who cant '
                                                                                  'stand neil', 'whats the dumbest '
                                                                                                'animal in the jungle'
                                                                                                ' a polar bear',
                 'im thinking about removing my spine i feel like its only holding me back', 'did you hear about the '
                                                                                             'two thieves who stole a '
                                                                                             'calendar they each got '
                                                                                             'six months',
                 'im terrified of elevators\xa0so im going to start taking steps to avoid them', 'have you heard of '
                                                                                                 'the band 923 '
                                                                                                 'megabytes probably '
                                                                                                 'not they havent had '
                                                                                                 'a gig yet',
                 'what do you call a psychic little person who has escaped from prison a small medium at large',
                 'the funniest dumb jokes your friends will adore', 'shutterstock', 'i used to hate facial '
                                                                                    'hair\xa0but then it grew on me',
                 'whats the difference between a dirty bus stop and a lobster with breast implants one is a crusty '
                 'bus station and the other is a busty crustacean', 'how many tickles does it take to make an octopus '
                                                                    'laugh ten tickles', 'i used to be addicted to '
                                                                                         'the hokey pokey\xa0but then '
                                                                                         'i turned myself around',
                 'whats the most terrifying word in nuclear physicsoops', 'i watched hockey before it was cool they '
                                                                          'were basically swimming', 'theres no hole '
                                                                                                     'in your shoe '
                                                                                                     'then howd you '
                                                                                                     'get your foot '
                                                                                                     'in it',
                 'a cowherd counted 48 cows on his property but when he rounded them up he had 50', 'when the two '
                                                                                                    'rabbit ears got '
                                                                                                    'married it was a '
                                                                                                    'nice ceremony '
                                                                                                    'but the '
                                                                                                    'reception '
                                                                                                    'was amazing',
                 'why couldnt the bicycle stand up because it was too tired', 'a chicken coup only has two doors if '
                                                                              'it had four it would be a chicken '
                                                                              'sedan', 'three fish are in a tank one '
                                                                                       'asks the others how do you '
                                                                                       'drive this thing',
                 'why dont crabs donate because theyre shellfish', 'what did blackbird say when he turned eightyaye '
                                                                   'matey', 'how does your feline shop by reading a '
                                                                            'catalogue', 'its hard to teach '
                                                                                         'kleptomaniacs humor they '
                                                                                         'take things so literally',
                 'sunnyside up scrambled or an omelet it doesnt matter theyre all eggcellent', 'dont worry if you '
                                                                                               'miss a gym session '
                                                                                               'everything will work '
                                                                                               'out', 'ever tried to '
                                                                                                      'eat a clock '
                                                                                                      'its '
                                                                                                      'timeconsuming', 'who can jump higher than a house pretty much anyone houses cant jump',
                 'what do an apple and an orange have in common neither one can drive',
                 'why did the businessman invest in smith  wollensky he wanted to stake his claim',
                 'five guys walk into a bar you think one of them wouldve seen it',
                 'what do you call a door when its not a door ajar',
                 'this sweet ride has four wheels and flies its a garbage truck',
                 'the most horrible jokes that will still make you chuckle', 'shutterstock',
                 'how many bugs do you need to rent out an apartment tenants',
                 'i want to go camping every year that trip was so in tents', 'wait you dont want to hear a joke '
                                                                              'about potassiumk',
                 'how do you organize a spacethemed hurrah you planet', 'your ex thats the punchline',
                 'how do you feel when theres no coffee depresso',
                 'i broke my arm in two places you know what the doctor told mestay out of those places',
                 'what do you give a sick bird tweetment',
                 'where did the king keep his armies up his sleevies', 'what are the biggest enemies of caterpillars '
                                                                       'dogerpillers',
                 'what do you call an empty can of cheese whiz cheese was',
                 'what did mario say when he broke up with princess peach its not you its ame mario',
                 'whats the award for being best dentist a little plaque',
                 'what did the finger say to the thumb im in glove with you',
                 'what do you call a magician dog a labracadabrador',
                 'what concert costs only 45 cents50 cent plus nickelback',
                 'what do sprinters eat before a race nothing they fast', 'who invented the round table sir cumference',
                 'what do you call the security guards outside of samsung the guardians of the galaxy', 'there are '
                                                                                                        'three types '
                                                                                                        'of people in '
                                                                                                        'the world '
                                                                                                        'those of us '
                                                                                                        'who are good '
                                                                                                        'at math and '
                                                                                                        'those of us '
                                                                                                        'who arent',
                 'what sound does a nut make when it sneezes cashew', 'why do ghosts love elevators because it lifts '
                                                                      'their spirits', 'whats the best way to carve '
                                                                                       'wood whittle by whittle',
                 'why was the snowman looking through a bag of carrots he was picking his nose', 'what do you call a '
                                                                                                 'belt made out of '
                                                                                                 'watches a waist of '
                                                                                                 'time',
                 'the best really bad jokes that are still hilarious', 'shutterstock', 'how can you make seven an '
                                                                                       'even number just take away '
                                                                                       'the s', 'what did the lawyer '
                                                                                                'wear to court a '
                                                                                                'lawsuit',
                 'what do you call hijklmnoh20', 'how do you find will smith in the snow just follow the fresh '
                                                 'prints', 'what is forrest gumps computer password1forrest1',
                 'what did the clock do when it was hungryit went back four seconds', 'what do you call a dog with no '
                                                                                      'legs you can call him whatever '
                                                                                      'you want hes still not '
                                                                                      'coming', 'i still remember the '
                                                                                                'last thing my '
                                                                                                'grandfather said '
                                                                                                'before kicking the '
                                                                                                'bucket hey you want '
                                                                                                'to see how far i can '
                                                                                                'kick this bucket',
                 'what do you call a can opener that doesnt work a cant opener', 'why did the man get fired from his '
                                                                                 'job at the calendar factory he took '
                                                                                 'a couple days off', 'why did the '
                                                                                                      'golfer wear '
                                                                                                      'two pairs of '
                                                                                                      'pants because '
                                                                                                      'he always gets '
                                                                                                      'a hole in '
                                                                                                      'one',
                 'did you hear about the kidnapping at school its fine he eventually woke up', 'what kind of dinosaur '
                                                                                               'loves to sleep well '
                                                                                               'now all of them',
                 'why did the teacher love the whiteboard she just thought it was remarkable', 'a guy told me nothing '
                                                                                               'rhymes with orangeso '
                                                                                               'i replied no it '
                                                                                               'doesnt',
                 'if youre american when you go in the bathroom and american when you come out what are you in the '
                 'bathroom european', 'whats red and bad for your teeth a brick', 'why cant a nose be 12 inches long '
                                                                                  'because then itd be a foot',
                 'whats the best part about living in switzerland im not sure but the flag is a big plus',
                 'did you hear the rumor about butter never mind i shouldnt spread it', 'what did the drummer call '
                                                                                        'his two twin daughters anna '
                                                                                        'one anna two', 'im not a big '
                                                                                                        'fan of '
                                                                                                        'stairs '
                                                                                                        'theyre '
                                                                                                        'always up to '
                                                                                                        'something',
                 'what do you call a boomerang that never comes back a stick', 'what to hear a joke about paper never '
                                                                               'mind its tearable', 'when is a joke a '
                                                                                                    'dad joke when '
                                                                                                    'its apparent'],
     'context_set': ''}]}

with open("data.json", "w") as outfile:
    json.dump(data, outfile)