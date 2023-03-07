import requests
from bs4 import BeautifulSoup
import regex as re


def scrape(url):
    page=requests.get(url)


    text=page.text
    text=text[text.index("<div id=\"mw-content-text\""):]
    text=text[:text.index("<span class=\"mw-headline\" id=\"References\">")]

    text=text[text.index("<p>"):]
    #print(page.text)
    #text=re.sub("<h3(?s).*</h3>", "", text)
    text=re.sub("<[^>]*>", "", text)

    results=""
    for line in text.split("\n"):
        if not re.match("\[.*?|.*?\]", line):
            results+=line+"\n"

    results=re.sub("&\#(.*?);(\:{0,1})(0|1|2|3|4|5|6|7|8|9){0,3}", "", results)
    #soup = BeautifulSoup(page.content, "html.parser")
    #results=soup.find(id="mw-content-text")

    return results
    #return results.prettify()


if __name__=="__main__":
    #scrape("https://simple.wikipedia.org/wiki/Dan_Kelly")
    print(scrape("https://simple.wikipedia.org/wiki/Dan_Kelly"))
    pass
