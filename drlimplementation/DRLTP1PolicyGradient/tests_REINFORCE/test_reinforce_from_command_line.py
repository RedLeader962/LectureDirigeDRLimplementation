# coding=utf-8
import os
import pytest

TARGET_WORKING_DIRECTORY = "DRLTP1PolicyGradient"

def set_up_cwd(initial_working_directory):
    print("\n:: START set_up_cwd, Initial was: ", initial_working_directory)

    if os.path.basename(initial_working_directory) != TARGET_WORKING_DIRECTORY:
        os.chdir("..")
        print(":: change cwd to: ", os.getcwd())
    return None


def return_to_initial_working_directory(initial_working_directory):
    print(":: return_to_initial_working_directory")
    if os.path.basename(os.getcwd()) != os.path.basename(initial_working_directory):
        os.chdir(initial_working_directory)
        print(":: change cwd to: ", os.getcwd())
    print(":: Teardown END\n")
    return None


@pytest.fixture(scope="function")
def set_up_PWD_to_project_root():
    initial_working_directory = os.getcwd()

    set_up_cwd(initial_working_directory)

    yield

    return_to_initial_working_directory(initial_working_directory)


def test_REINFORCE_agent_PLAY_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python REINFORCEplayingloop.py"
                 " --max_trj=6"
                 " --test_run=True")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


def test_REINFORCE_agent_TRAIN_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python REINFORCEtrainingloop.py"
                 " --render_env=False")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)