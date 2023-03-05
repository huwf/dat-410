from datetime import timedelta


class Weather:
    def __init__(self, **kwargs):
        pass


class Transport:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def answer(start, dest, time, leave=True, arrive_by=False):
        assert not all([leave, arrive_by]) and any([leave, arrive_by]), \
            "Must include exactly one of `leave` or `arrive_by` arguments"
        if leave:
            arrival = time + timedelta(minutes=36)
        else:
            arrival = time - timedelta(minutes=4)

        stops = [
            f"Take bus 25 from {start} to Sahlgrenska Huvudentre at {start + 6}",
            f"Change and take tram 6 to Järntorget",
            f"Take tram 3 to {dest}. You will arrive at {arrival}"
        ]
        return "\n".join(stops)


class Translation:
    def __init__(self, **kwargs):
        pass

