import collections

Transition = collections.namedtuple("Transition",
    [
        "obs",
        "action_idx",
        "advantage",
        "pts_my",
        "pts_opp",
        "value",
        "active_player",
        "log_prob",
        "is_forced",
        "imm_r",
        "meta_advantage",
    ],
    defaults=[0.0])

class Log:
    @staticmethod
    def info(msg):    print(f"\033[94m[INFO]\033[0m {msg}")
    @staticmethod
    def success(msg): print(f"\033[92m[SUCCESS]\033[0m {msg}")
    @staticmethod
    def warn(msg):    print(f"\033[93m[WARN]\033[0m {msg}")
    @staticmethod
    def error(msg):   print(f"\033[91m[ERROR]\033[0m {msg}")
    @staticmethod
    def phase(name):  print(f"\n\033[1;95m=== {name.upper()} ===\033[0m")
    @staticmethod
    def sim(msg, end="\n"): print(f"\r\033[96m[SIM]\033[0m {msg}", end=end, flush=True)
    @staticmethod
    def opt(msg, end="\n"): print(f"\r\033[93m[OPT]\033[0m {msg}", end=end, flush=True)
