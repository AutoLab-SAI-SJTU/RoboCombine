from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._GLOBAL_CONFIGS import asset_path

class hobby_desk_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        import glob as _glob
        disp_obj_list = ['073_rubikscube', '057_toycar', '079_remotecontrol']
        self.display_modelname = np.random.choice(disp_obj_list)
        _asset_path = str(asset_path('objects', self.display_modelname))
        _json_files = _glob.glob(os.path.join(_asset_path, 'model_data*.json'))
        _avail_ids = []
        for _f in _json_files:
            _b = os.path.basename(_f)
            try:
                _avail_ids.append(int(_b.replace('model_data', '').replace('.json', '')))
            except ValueError:
                pass
        if not _avail_ids:
            _avail_ids = [0]
        self.display_model_id = np.random.choice(_avail_ids)
        disp_obj_pose = rand_pose(xlim=[0.2, 0.28], ylim=[-0.05, 0.05], qpos=[0.707, 0.707, 0.0, 0.0], rotate_rand=True, rotate_lim=[0, np.pi / 3, 0])
        self.display_object = create_actor(scene=self, pose=disp_obj_pose, modelname=self.display_modelname, convex=True, model_id=self.display_model_id)
        self.display_object.set_mass(0.01)
        disp_stand_pose = rand_pose(xlim=[0.05, 0.1], ylim=[-0.15, -0.1], qpos=[0.707, 0.707, 0.0, 0.0], rotate_rand=True, rotate_lim=[0, np.pi / 6, 0])
        self.displaystand_id = np.random.choice([0, 1, 2, 3, 4])
        self.displaystand = create_actor(scene=self, pose=disp_stand_pose, modelname='074_displaystand', convex=True, model_id=self.displaystand_id)
        self.displaystand.set_mass(0.01)
        self.add_prohibit_area(self.displaystand, padding=0.05)
        self.add_prohibit_area(self.display_object, padding=0.1)
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
        print('[Subtask 1] Place object on stand')
        arm_tag_display = ArmTag('right')
        self.move(self.grasp_actor(self.display_object, arm_tag=arm_tag_display, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag_display, z=0.06))
        displaystand_fp = self.displaystand.get_functional_point(0)
        self.move(self.place_actor(self.display_object, arm_tag=arm_tag_display, target_pose=displaystand_fp, constrain='free', pre_dis=0.07))
        print('[Subtask 2] Move pill bottle to pad')
        arm_tag_pill = ArmTag('left')
        self.move(self.grasp_actor(self.pillbottle, arm_tag=arm_tag_pill, pre_grasp_dis=0.06, gripper_pos=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag_pill, z=0.05))
        pill_target = self.pill_pad.get_functional_point(1)
        self.move(self.place_actor(self.pillbottle, arm_tag=arm_tag_pill, target_pose=pill_target, pre_dis=0.05, dis=0, functional_point_id=0, pre_dis_axis='fp'))
        self.info['info'] = {'{A}': f'{self.display_modelname}/base{self.display_model_id}', '{B}': f'074_displaystand/base{self.displaystand_id}', '{a}': str(arm_tag_display), '{C}': f'080_pillbottle/base{self.pillbottle_id}', '{b}': str(arm_tag_pill)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            obj_pose = self.display_object.get_pose().p
            stand_pose = self.displaystand.get_pose().p
            eps1 = 0.03
            if np.all(abs(obj_pose[:2] - stand_pose[:2]) < np.array([eps1, eps1])):
                print('  Subtask 1 completed: Object placed on stand')
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
