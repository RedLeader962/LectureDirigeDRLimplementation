# coding=utf-8
import os
import pytest

ROOT_DIRECTORY = "DRLimplementation"
TARGET_WORKING_DIRECTORY = ROOT_DIRECTORY
# TARGET_WORKING_DIRECTORY = "SoftActorCritic"

def set_up_cwd(initial_CWD):
    print("\n:: START set_up_cwd, Initial was: ", initial_CWD)
    
    path_basename = os.path.basename(initial_CWD)
    
    if path_basename == TARGET_WORKING_DIRECTORY:
        pass
    elif path_basename == ROOT_DIRECTORY:
        # then we must get one level down in directory tree
        os.chdir(TARGET_WORKING_DIRECTORY)
        print(":: change cwd to: ", os.getcwd())
    else:
        # we are to deep in directory tree
        while os.path.basename(os.getcwd()) != TARGET_WORKING_DIRECTORY:
            os.chdir("..")
            print(":: CD to parent")
    return None


def return_to_initial_working_directory(initial_CWD):
    print(":: return_to_initial_working_directory")
    if os.path.basename(os.getcwd()) != os.path.basename(initial_CWD):
        os.chdir(initial_CWD)
        print(":: change cwd to: ", os.getcwd())
    print(":: Teardown END\n")
    return None


@pytest.fixture(scope="function")
def set_up_PWD_to_project_root():
    initial_CWD = os.getcwd()
    
    set_up_cwd(initial_CWD)
    
    yield
    
    return_to_initial_working_directory(initial_CWD)


#
# def test_SoftActorCritic_agent_PLAY_LUNAR_command_line_invocation(set_up_PWD_to_project_root):
#     from os import system
#
#     out = system("python -m SoftActorCritic --playLunar --play_for=6 --testRun")
#
#     # Note: exit(0) <==> clean exit without any errors/problems
#     assert 0 == out, "Agent invocated from command line exited with error {}".format(out)

# @pytest.mark.skip(reason="Mute for now")
def test_SoftActorCritic_agent_train_MountainCar_command_line_invocation(set_up_PWD_to_project_root):
    from os import system
    
    out = system("python -m SoftActorCritic --trainMontainCar --testRun")
    
    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


# @pytest.mark.skip(reason="Mute for now")
def test_SoftActorCritic_agent_train_Pendulum_command_line_invocation(set_up_PWD_to_project_root):
    from os import system
    
    out = system("python -m SoftActorCritic --trainPendulum --testRun")
    
    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


# @pytest.mark.skip(reason="Mute for now")
def test_SoftActorCritic_agent_train_LunarLander_command_line_invocation(set_up_PWD_to_project_root):
    from os import system
    
    out = system("python -m SoftActorCritic --trainLunarLander --testRun")
    
    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


# @pytest.mark.skip(reason="Problem solved")
def test_SoftActorCritic_agent_train_Pendulum_command_line_invocation_RERUN_PASS(set_up_PWD_to_project_root):
    from os import system
    
    out = system("python -m SoftActorCritic --trainPendulumTESTrerun --renderTraining --rerun 2")
    # out = system("python -m SoftActorCritic --trainPendulumTESTrerun --testRun --rerun 2")
    
    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)
