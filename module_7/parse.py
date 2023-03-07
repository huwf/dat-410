import re


class Parser:
    @staticmethod
    def find_topic(s):
        if re.match("^.*weather.*$|^.*temprature.*$", s):
            return True, Weather()
        if re.match("^.*translate.*$", s):
            return True, Translate()
        if re.match("^.*transport.*$|^.*get to.*$|^.*tram*$|^.*buss*", s):
            return True, Transport()

        print("Sorry i do not know what you mean!")
        return False, None

    def parse(self, str):
        topic_found, topic = self.find_topic(str)
        return topic_found, topic.find_args(str)


class Translate:
    word = ""
    from_language = ""
    to_language = ""

    def find_args(self, str):
        from_language = re.search(r'(?<=from )\w+', str)
        to_language = re.search(r'(?<=into )\w+ | (?<=to )\w+', str)
        word = re.search(r'(?<=what is )\w+ | (?<=translate )\w+', str)

        if not word:
            print("what word do you want to translate: ")
            self.word = input()
        else:
            self.word = word.group(0)

        if not from_language:
            print("What language do you want to translate it from?")
            self.from_language = input()
        else:
            self.from_language = from_language.group(0)

        if not to_language:
            print("what language do you want to translate it to:")
            self.to_language = input()
        else:
            self.to_language = to_language.group(0)

    def print_question(self):
        print("You want to translate" + self.word + " from " + self.from_language + " into" + self.to_language)


class Weather:
    location = ""
    time = "today"

    def find_args(self, str):
        location = re.search(r'(?<=in )\w+', str)
        time = re.search(r'(?<=for )\w+', str).group(0)

        # maybe the defult location could be "your location"
        if (self.location):
            print("where do you want to check the weather:")
            self.location = input()
        else:
            self.location = location.group(0)

    def print_question(self):
        print("you want to know what the weather is in " + self.location, "for " + self.time)


class Transport():
    start = ""
    dest = ""
    time = 0 

    def find_args(self, str):
        start = re.search(r'(?<=from )\w+', str)
        dest = re.search(r'(?<=to )\w+', str)
        time = re.search(r'(?<=at )\w+', str)
        

        if not start:
            print("where do you want to start:")
            self.start = input()
        else:
            self.start = start.group(0)

        if not dest:
            print("where do you want to go:")
            self.dest = input()
        else:
            self.dest = dest.group(0)

        if not time:
            print("when do you want to leave:")
            self.time = input() 
        else:
            self.time = time.group(0)   
