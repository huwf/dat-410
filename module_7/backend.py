import datetime
from datetime import timedelta


class Weather:
    @staticmethod
    def answer():
        pass


class Transport:

    @staticmethod
    def answer(from_location, to_location, time, arrive_by=True, leave=False):
        assert not all([leave, arrive_by]) and any([leave, arrive_by]), \
            "Must include exactly one of `leave` or `arrive_by` arguments"
        if isinstance(time, str):
            time = datetime.datetime.strptime(time, '%H:%M')
        if leave:
            arrival = time + timedelta(minutes=36)
        else:
            arrival = time - timedelta(minutes=4)

        to_str = f"Take tram 3 to {to_location}. " if to_location != 'Järntorget' else ""
        stops = [
            f"Take bus 25 from {from_location} to Sahlgrenska Huvudentre at {(time - timedelta(minutes=40)).time()}",
            f"At Sahlgrenska Huvudentre, take tram 6 to Järntorget",
            f"{to_str} You will arrive at {arrival.time()}"
        ]
        return "\n".join(stops)


class Translation:
    @staticmethod
    def answer(word, from_language, to_language):
        return f'{from_language} word {word} in {to_language} is antidisestablishmentarianism'

