from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class desk_arrange_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        stapler_pos = rand_pose(xlim=[0.15, 0.28], ylim=[-0.2, 0.0], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.stapler = create_actor(scene=self, pose=stapler_pos, modelname='048_stapler', convex=True, model_id=self.stapler_id)
        stapler_pad_pose = rand_pose(xlim=[0.15, 0.28], ylim=[-0.2, 0.0], qpos=[1, 0, 0, 0], rotate_rand=False)
        while np.sqrt((stapler_pad_pose.p[0] - stapler_pos.p[0]) ** 2 + (stapler_pad_pose.p[1] - stapler_pos.p[1]) ** 2) < 0.1:
            stapler_pad_pose = rand_pose(xlim=[0.15, 0.28], ylim=[-0.2, 0.0], qpos=[1, 0, 0, 0], rotate_rand=False)
        stapler_colors = {'Red': (1, 0, 0), 'Green': (0, 1, 0), 'Blue': (0, 0, 1), 'Yellow': (1, 1, 0), 'Cyan': (0, 1, 1), 'Magenta': (1, 0, 1), 'Black': (0, 0, 0), 'Gray': (0.5, 0.5, 0.5)}
        sc_items = list(stapler_colors.items())
        sc_idx = np.random.choice(len(sc_items))
        self.stapler_color_name, stapler_color_val = sc_items[sc_idx]
        self.stapler_pad = create_box(scene=self, pose=stapler_pad_pose, half_size=[0.055, 0.03, 0.0005], color=stapler_color_val, name='stapler_pad_right')
        self.add_prohibit_area(self.stapler, padding=0.1)
        self.add_prohibit_area(self.stapler_pad, padding=0.15)
        self.stapler_pad_pose = self.stapler_pad.get_pose().p.tolist() + [0.707, 0, 0, 0.707]
        ori_quat = [[0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]]
        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        phone_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.2, 0.0], qpos=ori_quat[self.phone_id], rotate_rand=True, rotate_lim=[0, 0.7, 0])
        self.phone = create_actor(scene=self, pose=phone_pose, modelname='077_phone', convex=True, model_id=self.phone_id)
        self.phone.set_mass(0.01)
        stand_pose = rand_pose(xlim=[-0.15, -0.05], ylim=[0, 0.2], qpos=[0.707, 0.707, 0, 0], rotate_rand=False)
        while np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2]) ** 2)) < 0.15:
            stand_pose = rand_pose(xlim=[-0.15, -0.05], ylim=[0, 0.2], qpos=[0.707, 0.707, 0, 0], rotate_rand=False)
        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.stand = create_actor(scene=self, pose=stand_pose, modelname='078_phonestand', convex=True, model_id=self.stand_id, is_static=True)
        self.add_prohibit_area(self.phone, padding=0.15)
        self.add_prohibit_area(self.stand, padding=0.15)

    def play_once(self):
        print('[Subtask 1] Move stapler to pad')
        arm_tag_stapler = ArmTag('right')
        self.move(self.grasp_actor(self.stapler, arm_tag=arm_tag_stapler, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag_stapler, z=0.1, move_axis='arm'))
        self.move(self.place_actor(self.stapler, target_pose=self.stapler_pad_pose, arm_tag=arm_tag_stapler, pre_dis=0.1, dis=0.0, constrain='align'))
        print('[Subtask 2] Place phone on stand')
        arm_tag_phone = ArmTag('left')
        self.move(self.grasp_actor(self.phone, arm_tag=arm_tag_phone, pre_grasp_dis=0.08))
        stand_func_pose = self.stand.get_functional_point(0)
        self.move(self.place_actor(self.phone, arm_tag=arm_tag_phone, target_pose=stand_func_pose, functional_point_id=0, dis=0, constrain='align'))
        self.info['info'] = {'{A}': f'048_stapler/base{self.stapler_id}', '{B}': self.stapler_color_name, '{a}': str(arm_tag_stapler), '{C}': f'077_phone/base{self.phone_id}', '{D}': f'078_phonestand/base{self.stand_id}', '{b}': str(arm_tag_phone)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            stapler_pose = self.stapler.get_pose().p
            stapler_qpose = np.abs(self.stapler.get_pose().q)
            target_pos = self.stapler_pad.get_pose().p
            eps = [0.02, 0.02, 0.01]
            pos_ok = np.all(abs(stapler_pose - target_pos) < np.array(eps))
            rot_ok = stapler_qpose.max() - stapler_qpose.min() < 0.02
            if pos_ok and rot_ok:
                print('  Subtask 1 completed: Stapler placed on pad')
                self.subtask_completed[0] = True
        if not self.subtask_completed[1]:
            phone_fp = np.array(self.phone.get_functional_point(0))
            stand_fp = np.array(self.stand.get_functional_point(0))
            eps = np.array([0.045, 0.04, 0.04])
            if np.all(np.abs(phone_fp - stand_fp)[:3] < eps):
                print('  Subtask 2 completed: Phone placed on stand')
                self.subtask_completed[1] = True
        left_gripper_val = self.robot.left_gripper_val
        right_gripper_val = self.robot.right_gripper_val
        gripper_check_mode = getattr(self, 'gripper_check_mode', 'full')
        if gripper_check_mode == 'none':
            left_open = True
            right_open = True
            gripper_open = True
        elif gripper_check_mode == 'relaxed':
            left_open = left_gripper_val > 0.6
            right_open = right_gripper_val > 0.05
            gripper_open = left_open and right_open
        else:
            left_open = self.robot.is_left_gripper_open()
            right_open = self.robot.is_right_gripper_open()
            gripper_open = left_open and right_open
        current_step = self.take_action_cnt
        success = all(self.subtask_completed) and gripper_open
        if not hasattr(self, '_last_logged_step'):
            self._last_logged_step = -1
        if current_step != self._last_logged_step:
            self._last_logged_step = current_step
            mode_indicator = {'full': '', 'relaxed': '[R]', 'none': '[N]'}[gripper_check_mode]
            print(f'[Step {current_step:4d}]{mode_indicator} S1:{(chr(10003) if self.subtask_completed[0] else chr(10007))} S2:{(chr(10003) if self.subtask_completed[1] else chr(10007))} L:{left_gripper_val:.3f}({(chr(10003) if left_open else chr(10007))}) R:{right_gripper_val:.3f}({(chr(10003) if right_open else chr(10007))}) -> ', end='')
            if success:
                print('SUCCESS!')
            else:
                reasons = []
                if not self.subtask_completed[0]:
                    reasons.append('S1')
                if not self.subtask_completed[1]:
                    reasons.append('S2')
                if all(self.subtask_completed) and gripper_check_mode != 'none':
                    if not left_open:
                        reasons.append(f"L<{(0.6 if gripper_check_mode == 'relaxed' else 0.8)}")
                    if not right_open:
                        reasons.append(f"R<{(0.05 if gripper_check_mode == 'relaxed' else 0.8)}")
                print(f"FAIL: {','.join(reasons)}")
        return success
