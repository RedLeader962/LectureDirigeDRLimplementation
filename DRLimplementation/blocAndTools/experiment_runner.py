# coding=utf-8
from typing import Tuple, List, Any

from blocAndTools import ExperimentSpec, Agent
from argparse import Namespace

def run_experiment(hparam: dict, args_: Namespace, test_hparam, rerun_nb=1) -> Tuple[dict, str, list]:
    """
    Set up and execute experirment
    Is responsible of:
        1- instantiate a ExperiementSpec
        2- check for dictionary field with list of values ex: 'theta_nn_h_layer_topo': [(16, 32), (64, 64), (84, 84)],
        2- setup experiment for hyperparameter search over those values
        3- execute training following experiment specification
        4- execute rerun as requested
        5- report result in consol message

    :param hparam: a dict of hyperparameter and experiment specification
    :param rerun_nb: the number of requested rerun
    :param args_: a argument parser Namespace
    :param test_hparam: a dict of test hyperparameter and test experiment specification
    :return: init_hparam, the hparam field key with multiple values to experiment, it's values_search_set
    """
    if args_.testRun:
        hparam = test_hparam

    init_hparam = hparam.copy()

    exp_spec = ExperimentSpec()

    hparam_search_list, key, values_search_set = configure_experiment_hparam_search(hparam)

    exp_hparam_search_str = values_search_list_to_regex_compatible_str(key, values_search_set)
    exp_rerun_tag = init_hparam['rerun_tag']
    exp_rerun_tag = exp_rerun_tag + '-' + exp_hparam_search_str
    print(":: TensorBoard rerun tag: {}\n".format(exp_rerun_tag),)

    for hparam in hparam_search_list:
        for run_idx in range(rerun_nb):
            print(":: Starting rerun experiment no {}".format(run_idx))
            exp_spec = prep_exp_spec_for_run(hparam, run_idx, args_=args_, exp_spec=exp_spec)
            warmup_agent_for_training(exp_spec, args_)

    return init_hparam, key, values_search_set

def configure_experiment_hparam_search(hparam: dict) -> Tuple[List[dict], Any, Any]:
    """
    Build a list of hyperparameter dict
        1- search the hparam dict for a field with a list of values;
        2- if their is one, build a list of hyperparameter dict with each one taking a unique value
            from that field values list;

    :param hparam: a dict of hyperparameter and experiment specification
    :return: a list of configure hparam and the ongoing hyperparameter search  key, values
    """
    aKey, values = test_hparam_search_set(hparam)
    if aKey is None:
        return [hparam], None, None
    else:
        hparam_search_list = []
        for v in values:
            hp = hparam.copy()
            hp[aKey] = v
            tag = hp['rerun_tag']
            value_str = str(v)
            value_str = value_str.replace(' ', '')
            hp['rerun_tag'] = tag + '-' + aKey + '=' + value_str
            hparam_search_list.append(hp)
        return hparam_search_list, aKey, values


def prep_exp_spec_for_run(hparam: dict, run_idx, args_, exp_spec) -> ExperimentSpec:
    exp_spec.set_experiment_spec(hparam)

    exp_spec.rerun_idx = run_idx

    if args_.discounted is not None:
        exp_spec.set_experiment_spec({'discounted_reward_to_go': args_.discounted})

    return exp_spec


def values_search_list_to_regex_compatible_str(akey, aValues_search_list) -> str:
    """
    Convert a values search list to a regex compatible search string
    :param akey: a dictionary key
    :param aValues_search_list: a list of dictionary values
    :return: a regex compatible string
    """
    if akey is None:
        return ''
    else:
        values_str = ''
        for v in aValues_search_list:
            v_str = str(v)
            v_str = v_str.replace(' ', '')
            v_str = v_str.replace('(', '\\(')
            v_str = v_str.replace(')', '\\)')
            values_str += v_str + '|'

        values_str = values_str.rstrip('|')
        hparam_search_str = akey + '=(' + values_str + ')'
        return hparam_search_str


def test_hparam_search_set(hparam: dict) -> Tuple[str, list] or Tuple[None, None]:
    """
    Test if a specifiation dictionary as a field containing multiple values to experiment
    :param hparam: the specification for an experiment
    :return: the field key and values_search_set to execute or None if all field are single value
    """
    for k, values in hparam.items():
        if isinstance(values, list):
            return k, values
    return None, None


def warmup_agent_for_training(spec: ExperimentSpec, args_: Namespace) -> None:
    agent = spec['AgentType']
    ac_agent: Agent = agent(spec)
    ac_agent.train(render_env=args_.renderTraining)
    ac_agent.__del__()


def warmup_agent_for_playing(run_name, spec: ExperimentSpec, args_: Namespace):
    agent = spec['AgentType']
    ac_agent: Agent = agent(spec)
    ac_agent.play(run_name=run_name, max_trajectories=args_.play_for)


def experiment_start_message(consol_width, rerun_nb) -> None:
    print("\n")
    for _ in range(3):
        print("\\" * consol_width)

    print("\n:: The experiment will be rerun {} time".format(rerun_nb))


def experiment_closing_message(initial_hparam, nb_of_rerun, key, values_search_set, consol_width) -> None:
    name = initial_hparam['paramameter_set_name']
    name += " " + initial_hparam['comment']
    exp_rerun_tag = initial_hparam['rerun_tag']
    if key is None:
        print("\n:: The experiment - {} - was rerun {} time".format(name, nb_of_rerun),
              "\n:: TensorBoard rerun tag: {}".format(exp_rerun_tag),
              "\n")
    else:
        exp_hparam_search_str = values_search_list_to_regex_compatible_str(key, values_search_set)
        exp_rerun_tag = exp_rerun_tag + '-' + exp_hparam_search_str

        initial_hparam['rerun_tag'] = exp_rerun_tag
        exp_spec = ExperimentSpec()
        exp_spec.set_experiment_spec(initial_hparam, print_change=False)

        print("\n:: Experiment - {}:\n\n".format(name),
              exp_spec.__repr__(),
              "\n\n:: Experiment"
              "\n       ↳ was run over hparam name[v, ...]: {}{}".format(key, str(values_search_set)),
              "\n       ↳ each 'values' was rerun {} time".format(nb_of_rerun),
              "\n       ↳ for a grand total of {} runs".format(nb_of_rerun * len(values_search_set)),
              "\n\n:: TensorBoard rerun tag: {}\n".format(exp_rerun_tag),
              )

    for _ in range(3):
        print("/" * consol_width)
