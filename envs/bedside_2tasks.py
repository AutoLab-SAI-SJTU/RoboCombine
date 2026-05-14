from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class bedside_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        fan_pose = rand_pose(xlim=[0.05, 0.1], ylim=[-0.15, -0.05], qpos=[0.0, 0.0, 0.707, 0.707], rotate_rand=True, rotate_lim=[0, 2 * np.pi, 0])
        self.fan_id = np.random.choice([4, 5])
        self.fan = create_actor(scene=self, pose=fan_pose, modelname='099_fan', convex=True, model_id=self.fan_id)
        self.fan.set_mass(0.01)
        fan_pad_pose = rand_pose(xlim=[0.15, 0.25], ylim=[-0.15, -0.05])
        fan_colors = {'Red': (1, 0, 0), 'Green': (0, 1, 0), 'Blue': (0, 0, 1), 'Yellow': (1, 1, 0), 'Orange': (1, 0.5, 0)}
        fc_items = list(fan_colors.items())
        fc_idx = np.random.choice(len(fc_items))
        self.fan_color_name, fan_color_val = fc_items[fc_idx]
        self.fan_pad = create_box(scene=self, pose=fan_pad_pose, half_size=(0.05, 0.05, 0.001), color=fan_color_val, name='fan_pad_right')
        self.fan_pad.set_mass(1)
        self.add_prohibit_area(self.fan, padding=0.07)
        self.prohibited_area.append([fan_pad_pose.p[0] - 0.15, fan_pad_pose.p[1] - 0.15, fan_pad_pose.p[0] + 0.15, fan_pad_pose.p[1] + 0.15])
        self.fan_target_pose = self.fan_pad.get_pose().p.tolist() + [1, 0, 0, 0]
        pill_pos = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.1, 0.1], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False)
        self.pillbottle_id = np.random.choice([1, 2, 3, 4, 5], 1)[0]
        self.pillbottle = create_actor(scene=self, pose=pill_pos, modelname='080_pillbottle', convex=True, model_id=self.pillbottle_id)
        self.pillbottle.set_mass(0.05)
        pill_tgt_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.1, 0.1], qpos=[1, 0, 0, 0], rotate_rand=False)
        while np.sqrt((pill_tgt_pose.p[0] - pill_pos.p[0]) ** 2 + (pill_tgt_pose.p[1] - pill_pos.p[1]) ** 2) < 0.1:
            pill_tgt_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.1, 0.1], qpos=[1, 0, 0, 0], rotate_rand=False)
        self.pill_pad = create_box(scene=self, pose=pill_tgt_pose, half_size=[0.04, 0.04, 0.0005], color=(0, 0, 1), name='pillbottle_pad_left', is_static=True)
        self.add_prohibit_area(self.pillbottle, padding=0.05)
        self.add_prohibit_area(self.pill_pad, padding=0.1)

    def play_once(self):
        print('[Subtask 1] Place fan on pad')
        arm_tag_fan = ArmTag('right')
        self.move(self.grasp_actor(self.fan, arm_tag=arm_tag_fan, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag_fan, z=0.05))
        self.move(self.place_actor(self.fan, arm_tag=arm_tag_fan, target_pose=self.fan_target_pose, constrain='align', pre_dis=0.04, dis=0.005))
        print('[Subtask 2] Move pill bottle to pad')
        arm_tag_pill = ArmTag('left')
        self.move(self.grasp_actor(self.pillbottle, arm_tag=arm_tag_pill, pre_grasp_dis=0.06, gripper_pos=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag_pill, z=0.05))
        pill_target = self.pill_pad.get_functional_point(1)
        self.move(self.place_actor(self.pillbottle, arm_tag=arm_tag_pill, target_pose=pill_target, pre_dis=0.05, dis=0, functional_point_id=0, pre_dis_axis='fp'))
        self.info['info'] = {'{A}': f'099_fan/base{self.fan_id}', '{B}': self.fan_color_name, '{a}': str(arm_tag_fan), '{C}': f'080_pillbottle/base{self.pillbottle_id}', '{b}': str(arm_tag_pill)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            fan_qpose = self.fan.get_pose().q
            fan_pose = self.fan.get_pose().p
            target_pose = self.fan_target_pose[:3]
            target_qpose = np.array([0.707, 0.707, 0.0, 0.0])
            if fan_qpose[0] < 0:
                fan_qpose = fan_qpose * -1
            eps = np.array([0.05, 0.05, 0.05, 0.05])
            rot_ok = np.all(abs(fan_qpose - target_qpose) < eps)
            pos_ok = np.all(abs(fan_pose - target_pose) < np.array([0.04, 0.04, 0.04]))
            if rot_ok and pos_ok:
                print('  Subtask 1 completed: Fan placed on pad')
                self.subtask_completed[0] = True
        if not self.subtask_completed[1]:
            pillbottle_pos = self.pillbottle.get_pose().p
            target_pos = self.pill_pad.get_pose().p
            eps1 = 0.03
            pos_ok = np.all(abs(pillbottle_pos[:2] - target_pos[:2]) < np.array([eps1, eps1]))
            z_ok = np.abs(self.pillbottle.get_pose().p[2] - (0.741 + self.table_z_bias)) < 0.005
            if pos_ok and z_ok:
                print('  Subtask 2 completed: Pill bottle placed on pad')
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
