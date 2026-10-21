"""Map the original human trajectory to MJCFs with passive interface joints."""
import numpy as np


class HumanStateMap:
    def __init__(self, model):
        self.model = model
        qpos, qvel, travel_indices, travel_limits = [], [], [], []
        self.hip_copies = []
        for j in range(model.njnt):
            joint = model.joint(j)
            name = joint.name or ''
            if name.startswith('interface_'):
                if '_slide_' in name or '_hinge_' in name:
                    travel_indices.append(int(joint.qposadr[0]))
                    travel_limits.append(float(np.max(np.abs(joint.range))))
                if name.startswith('interface_hip_'):
                    human = model.joint(name.removeprefix('interface_'))
                    self.hip_copies.append((int(joint.qposadr[0]), int(human.qposadr[0]),
                                            int(joint.dofadr[0]), int(human.dofadr[0])))
                continue
            # Free=0, ball=1, slide=2, hinge=3 in mjtJoint.
            kind = int(joint.type[0])
            nq, nv = (7, 6) if kind == 0 else (4, 3) if kind == 1 else (1, 1)
            qpos.extend(range(int(joint.qposadr[0]), int(joint.qposadr[0]) + nq))
            qvel.extend(range(int(joint.dofadr[0]), int(joint.dofadr[0]) + nv))
        self.qpos_indices = np.asarray(qpos, dtype=np.intp)
        self.qvel_indices = np.asarray(qvel, dtype=np.intp)
        self.interface_qpos_indices = np.setdiff1d(np.arange(model.nq), self.qpos_indices)
        self.interface_qvel_indices = np.setdiff1d(np.arange(model.nv), self.qvel_indices)
        self.interface_state_dim = self.interface_qpos_indices.size + self.interface_qvel_indices.size
        self.travel_indices = np.asarray(travel_indices, dtype=np.intp)
        self.travel_limits = np.asarray(travel_limits)

    def interface_state(self, data):
        """Critic-only state: interface coordinates, then their velocities."""
        return np.concatenate((data.qpos[self.interface_qpos_indices],
                               data.qvel[self.interface_qvel_indices])).astype(np.float32)

    def travel_fraction(self, qpos):
        """Largest cuff travel / emergency stop range; 0 on legacy models."""
        if not self.travel_indices.size:
            return 0.
        return float(np.max(np.abs(qpos[self.travel_indices]) / self.travel_limits))

    def expand(self, human_qpos, human_qvel=None):
        human_qpos = np.asarray(human_qpos, dtype=np.float64)
        if human_qpos.shape != self.qpos_indices.shape:
            raise ValueError(f'Human qpos shape {human_qpos.shape} != {self.qpos_indices.shape}')
        qpos = self.model.qpos0.copy()
        qpos[self.qpos_indices] = human_qpos
        qvel = None
        if human_qvel is not None:
            human_qvel = np.asarray(human_qvel, dtype=np.float64)
            if human_qvel.shape != self.qvel_indices.shape:
                raise ValueError(f'Human qvel shape {human_qvel.shape} != {self.qvel_indices.shape}')
            qvel = np.zeros(self.model.nv)
            qvel[self.qvel_indices] = human_qvel
        # Start the robot hip at the reference hip angle/rate, retaining the
        # geometric centre offset. Local cuff translations/rotations start at 0.
        for eq, hq, ev, hv in self.hip_copies:
            qpos[eq] = qpos[hq]
            if qvel is not None:
                qvel[ev] = qvel[hv]
        return qpos, qvel
