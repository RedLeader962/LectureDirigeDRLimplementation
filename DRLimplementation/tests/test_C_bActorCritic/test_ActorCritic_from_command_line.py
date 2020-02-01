# coding=utf-8
import os
import pytest

ROOT_DIRECTORY = "DRLimplementation"
TARGET_WORKING_DIRECTORY = ROOT_DIRECTORY


# TARGET_WORKING_DIRECTORY = "ActorCritic"

# pytestmark = pytest.mark.skip("all tests still WIP")  # (Priority) todo:implement --> coverage: then remove line

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


def test_ActorCritic_agent_discrete_PLAY_CARTPOLE_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --playCartpole --play_for=6 --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)

def test_ActorCritic_agent_discrete_PLAY_LUNAR_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --playLunar --play_for=6 --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


def test_ActorCritic_agent_discrete_TRAIN_BATCH_SPLIT_MONTECARLO_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainSplitMC --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)

def test_ActorCritic_agent_discrete_TRAIN_BATCH_SPLIT_BOOTSTRAP_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainSplitBootstrap --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)

def test_ActorCritic_agent_discrete_TRAIN_BATCH_SHARED_BOOTSTRAP_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainSharedBootstrap --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


def test_ActorCritic_agent_discrete_TRAIN_ONLINE_SPLIT_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainOnlineSplit --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)

def test_ActorCritic_agent_discrete_TRAIN_ONLINE_SHARED_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainOnlineShared3layer --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


def test_ActorCritic_agent_discrete_TRAIN_ONLINE_SPLIT_TWO_INPUT_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainOnlineSplitTwoInputAdvantage --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)


def test_ActorCritic_agent_discrete_TRAIN_BATCH_SHARED_command_line_invocation(set_up_PWD_to_project_root):
    from os import system

    out = system("python -m ActorCritic --trainSharedBootstrap --testRun")

    # Note: exit(0) <==> clean exit without any errors/problems
    assert 0 == out, "Agent invocated from command line exited with error {}".format(out)



