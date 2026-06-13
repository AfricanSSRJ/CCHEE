from .io_utils import load_json


class TypeSystem:
    def __init__(self, schema_path):
        self.schema = load_json(schema_path)
        self.type_roles = {}
        self.coarse_to_fine = {}
        self.pair_to_full = {}
        self.full_to_pair = {}
        roles = set()
        for item in self.schema:
            full = item["event_type"]
            coarse = full.split("-")[0]
            fine = full.split("-")[-1]
            self.type_roles[full] = [r["role"] for r in item["role_list"]]
            self.coarse_to_fine.setdefault(coarse, set()).add(fine)
            self.pair_to_full[(coarse, fine)] = full
            self.full_to_pair[full] = (coarse, fine)
            roles.update(self.type_roles[full])
        self.fine_types = list(self.type_roles.keys())
        self.coarse_types = sorted(self.coarse_to_fine.keys())
        self.roles = sorted(roles)
        self.role2id = {"O": 0}
        for r in self.roles:
            self.role2id[r] = len(self.role2id)
        self.id2role = {v: k for k, v in self.role2id.items()}

    def coarse_of(self, full):
        return self.full_to_pair[full][0]

    def fine_of(self, full):
        return self.full_to_pair[full][1]

    def roles_of(self, full):
        return self.type_roles.get(full, [])

    def is_legal_pair(self, coarse, fine):
        return fine in self.coarse_to_fine.get(coarse, set())

    def to_full(self, coarse, fine):
        return self.pair_to_full.get((coarse, fine))

    def known_type(self, full):
        return full in self.full_to_pair
