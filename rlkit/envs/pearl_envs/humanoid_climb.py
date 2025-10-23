import numpy as np
import trimesh
import mujoco
import time
from gym.envs.mujoco import HumanoidEnv as HumanoidEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class HumanoidClimbEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=1, randomize_tasks=False):
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        self.active_contact_dim=4
        self.deactive_contact_dim=1
        self.contact_dim=self.active_contact_dim*self.deactive_contact_dim
        
        super(HumanoidClimbEnv, self).__init__()
    

    def step(self,obs,action,contact):#contact: [active_contact_dim]
        #time.sleep(0.03)
        if abs(obs[0])>1e-4:
            contact=obs[1:5]
        #c=np.argmax(contact.reshape(-1,2), axis=1)
        c = (contact > 0).astype(int)
        qpos = self.sim.data.qpos
        pos_before = np.copy(mass_center(self.model, self.sim)[::2])
        self.do_simulation(action,self.frame_skip,c)
        pos_after = np.array([qpos[0],qpos[2]])#mass_center(self.model, self.sim)[0::2]

        alive_bonus = 11.28
        fall = 15.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal), np.sin(self._goal))
        goal_pos = np.array([3.1,6.28])
        lin_vel_cost = 1.5 * np.sum((pos_after[1] - pos_before[1])) / 0.015
        lin_pos_cost = -0.4 * np.sum(pow(goal_pos-pos_after,2))
        quad_ctrl_cost = 0.002 * np.square(data.ctrl).sum()
        contact_cost=-0.1*np.sum(c)
        quad_impact_cost = 2.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = contact_cost+lin_pos_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        #reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + 5
        #reward = 1.0 * np.sum(pow(pos_after[1],2))
        done = bool(qpos[2] < 0.9) 
        next_obs=self._get_obs()
        if abs(obs[0])<1e-4:
            next_obs[0]=1.0
            next_obs[1:5]=contact
        else:
            next_obs[:5]=obs[:5]
            next_obs[0]-=0.1
        if(qpos[2]>1.6): print(qpos[2])
        return next_obs, reward, done, dict(reward_linvel=lin_pos_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([[0.,0.,0.,0.,0.,],
                               data.qpos.flat,
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        
    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': d} for d in directions]
        return tasks
    
    def compute_distance(self, capsule_id, box_id):
        """
        计算盒子到药丸的最近距离。
        """
        # 获取药丸体属性
        capsule_pos = self.data.geom_xpos[capsule_id]  # 药丸体位置
        capsule_mat = self.data.geom_xmat[capsule_id].reshape(3, 3)  # 药丸体旋转矩阵
        capsule_size = self.model.geom_size[capsule_id]  # [radius, half-length]

        # 获取盒子属性
        box_pos = self.data.geom_xpos[box_id]  # 盒子位置
        box_mat = self.data.geom_xmat[box_id].reshape(3, 3)  # 盒子旋转矩阵
        box_size = self.model.geom_size[box_id]  # [half-x, half-y, half-z]

        # 计算药丸体的轴线两端点（线段）
        axis = capsule_mat[:, 2]  # 药丸体的 z 轴方向
        capsule_start = capsule_pos - capsule_size[1] * axis  # 圆柱体一端
        capsule_end = capsule_pos + capsule_size[1] * axis  # 圆柱体另一端

        # 分别计算盒子到药丸首位和中间的距离
        distance_sphere1 = self._point_to_box_distance(capsule_start, box_pos, box_mat, box_size) - capsule_size[0]
        distance_sphere2 = self._point_to_box_distance(capsule_end, box_pos, box_mat, box_size) - capsule_size[0]
        distance_line = self._line_to_box_distance(capsule_start, capsule_end, box_pos, box_mat, box_size) - capsule_size[0]
        # 返回最小值
        return min(distance_sphere1, distance_sphere2, distance_line)

    def _point_to_box_distance(self, point, box_pos, box_mat, box_size):
        """
        计算点到盒子的最近距离。
        """
        # 将点转换到盒子的局部坐标系
        local_point = np.dot(point - box_pos, box_mat.T)

        # 裁剪点到盒子表面
        clamped_point = np.maximum(np.minimum(local_point, box_size), -box_size)

        # 转回全局坐标系
        closest_point = np.dot(clamped_point, box_mat) + box_pos

        # 计算点到最近点的欧几里得距离
        return np.linalg.norm(point - closest_point)

    def _line_to_box_distance(self, start, end, box_pos, box_mat, box_size):
        """
        精确计算线段到盒子的最近距离，考虑线段与盒子表面的交点和端点到盒子的距离。
        """
        # 将线段的起点和终点转换到盒子局部坐标系
        local_start = np.dot(start - box_pos, box_mat.T)
        local_end = np.dot(end - box_pos, box_mat.T)

        # 初始化最近距离为无穷大
        closest_distance = float('inf')

        # 遍历盒子的 6 个面（3 个轴的正负方向）
        for i in range(3):
            # 当前轴的下界和上界
            lower_bound = -box_size[i]
            upper_bound = box_size[i]

            # 判断线段是否与当前轴的平面相交
            for bound in [lower_bound, upper_bound]:
                # 计算线段在当前轴上的 t 值（参数化线段方程）
                t = (bound - local_start[i]) / (local_end[i] - local_start[i]) if local_end[i] != local_start[i] else None

                if t is not None and 0 <= t <= 1:
                    # 计算交点在局部坐标系中的位置
                    intersection = local_start + t * (local_end - local_start)

                    # 检查交点是否在盒子的其他两个轴方向的边界内
                    if (
                        -box_size[(i + 1) % 3] <= intersection[(i + 1) % 3] <= box_size[(i + 1) % 3] and
                        -box_size[(i + 2) % 3] <= intersection[(i + 2) % 3] <= box_size[(i + 2) % 3]
                    ):
                        # 计算交点到线段的距离
                        closest_distance = min(closest_distance, 0)  # 在盒子内，直接距离为 0

            # 检查线段端点到盒子的最小距离
            for point in [local_start, local_end]:
                clamped_point = np.maximum(np.minimum(point, box_size), -box_size)
                distance = np.linalg.norm(point - clamped_point)
                closest_distance = min(closest_distance, distance)

        return closest_distance