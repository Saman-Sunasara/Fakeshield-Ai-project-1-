import pandas as pd
import os

os.makedirs('dataset', exist_ok=True)

true_data = {
    'title': [
        'U.S. military to accept transgender recruits on Monday: Pentagon',
        'Senior U.S. Republican senator: Let Mr. Mueller do his job',
        'FBI Russia probe helped by Australian diplomat tip-off: NYT',
        'Trump wants Postal Service to charge much more for Amazon shipments',
        'White House, Congress prepare for tax bill negotiations'
    ] * 10,
    'text': [
        'WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday...',
        'WASHINGTON (Reuters) - The top Republican on the U.S. Senate Armed Services Committee said on Sunday that special counsel Robert Mueller...',
        'WASHINGTON (Reuters) - Trump campaign adviser George Papadopoulos told an Australian diplomat in May 2016 that Russia had political dirt...',
        'WASHINGTON (Reuters) - President Donald Trump called on the U.S. Postal Service on Friday to charge much more to ship packages for Amazon...',
        'WASHINGTON (Reuters) - Trump administration officials and lawmakers are gearing up for a tough negotiation over the tax reform bill...'
    ] * 10,
    'subject': ['politicsNews'] * 50,
    'date': ['December 29, 2017', 'December 31, 2017', 'December 30, 2017', 'December 29, 2017', 'October 25, 2017'] * 10
}

fake_data = {
    'title': [
        'Donald Trump Sends Embarrassing New Year’s Eve Message',
        'Drunk Bragging Trump Staffer Started Russian Collusion Investigation',
        'Sheriff David Clarke Becomes An Internet Joke For Claiming He Was \'Cleared\'',
        'Trump Is So Obsessed He Even Has Obama’s Name Coded Into His Website',
        'Pope Francis Just Called Out Donald Trump During His Christmas Speech'
    ] * 10,
    'text': [
        'Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies...',
        'House Intelligence Committee Chairman Devin Nunes is going to have a bad day. He s been under the assumption that the dossier...',
        'On Friday, it was revealed that former Milwaukee Sheriff David Clarke, who was being considered for Homeland Security Secretary...',
        'On Christmas day, Donald Trump announced that he would be back to work the following day, but he is golfing for the fourth day in a row...',
        'Pope Francis used his annual Christmas Day message to rebuke Donald Trump without even calling him by his name...'
    ] * 10,
    'subject': ['News'] * 50,
    'date': ['December 31, 2017', 'December 31, 2017', 'December 30, 2017', 'December 29, 2017', 'December 25, 2017'] * 10
}

pd.DataFrame(true_data).to_csv('dataset/True.csv', index=False)
pd.DataFrame(fake_data).to_csv('dataset/Fake.csv', index=False)
print("Dataset created successfully.")
