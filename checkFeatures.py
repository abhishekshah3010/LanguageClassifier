"""
file: checkFeatures.py
description: CSCI 630, Lab 3, Wikipedia Language Classification
language: Python
author: Abhishek Shah, as5553
"""

def commonDutchWords(statement):
    """
    Checking for the presence of common dutch words
    """
    commonWords = ['naar', 'be', 'en', 'ik', 'het', 'voor', 'niet', 'met', 'hij', 'zijn', 'ze', 'wij', 'ze', 'er',
                   'hun', 'de',
                   'zo', 'over', 'hem', 'weten', 'jouw', 'dan', 'ook', 'onze', 'deze', 'ons', 'meest', 'niet', 'wat',
                   'ze', 'zijn', 'maar', 'die', 'heb', 'voor', 'ben', 'mijn', 'dit', 'hem', 'hebben', 'heeft', 'nu',
                   'allemaal', 'gedaan', 'huis', 'zij', 'jaar', 'vader', 'doet', 'vrouw', 'geld', 'hun', 'anders',
                   'zitten', 'niemand', 'binnen', 'spijt', 'maak', 'staat', 'werk', 'moeder', 'gezien', 'waren',
                   'wilde', 'praten', 'genoeg', 'meneer', 'klaar', 'ziet', 'een']
    words = statement.split()
    for word in words:
        if word.lower().replace(',', '') in commonWords:
            return True
    return False


def commonEnglishWords(statement):
    """
    Checking for the presence of common english words
    """
    commonWords = ['to', 'be', 'I', 'it', 'for', 'not', 'with', 'he', 'his', 'they', 'we', 'she', 'there', 'their',
                   'so', 'about', 'me', 'the', 'of', 'have', 'him', 'know', 'your', 'than', 'then', 'also', 'our',
                   'these', 'us', 'most', 'but', 'and']
    words = statement.split()
    for word in words:
        if word.lower().replace(',', '') in commonWords:
            return True
    return False


def englishArticles(statement):
    """
    Check for the presence of articles a an the
    If they are present , chances are statement is in english language
    """
    allWords = statement.split()
    for word in allWords:
        if word.lower().replace(',', '') == 'a' or word.lower().replace(',', '') == 'an' or \
                word.lower().replace(',', '') == 'the':
            return True
    return False


def stringVan(statement):
    """
    Check if the statement contains the string van
    """
    allWords = statement.split()
    for word in allWords:
        if word.lower().replace(',', '') == 'van':
            return True
    return False


def stringDeHet(statement):
    """
    Check if the statement contains the string de and het
    """
    allWords = statement.split()
    for word in allWords:
        if word.lower().replace(',', '') == 'de' or word.lower().replace(',', '') == 'het':
            return True
    return False
