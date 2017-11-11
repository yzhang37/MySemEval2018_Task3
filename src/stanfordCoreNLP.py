import json, requests
import pprint


# http://stanfordnlp.github.io/CoreNLP/ssplit.html

class StanfordCoreNLP(object):

    def __init__(self, server_url):
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url

    def annotate(self, text, properties=None):

        text = text.encode("utf-8", errors="ignore")

        if not properties:
            properties = {}
        r = requests.get(
            self.server_url, params={
                'properties': str(properties)
            }, data=text)
        output = r.text

        if ('outputFormat' in properties
             and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, strict=False)
            except:
                pass
        return output

    # Only split sentences on newlines, only separating words on whitespace.
    def parse_one_line(self, line):
        line = line.strip()
        if line == "":
            return None
        # try:
        output = self.annotate(line, properties={
            'ssplit.eolonly': "false",  # Only split sentences on newlines
            'tokenize.whitespace': "true",  # only separating words on whitespace.
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse',
            'outputFormat': 'json'
        })
        return output
if __name__ == '__main__':

    server_url = 'http://precision:9000'
    nlp = StanfordCoreNLP(server_url)

    result = nlp.parse_one_line("@j0nathandavis They who? Stupid and partial opinions like this one only add noise to any debate.")
    json.dump(result,open("parser_text.json","w"),indent=2)
    # s = "Elon Musk \u00e2 "
    # print(s)
    # print(result)

