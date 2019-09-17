# !/usr/bin/env python

def environnement_doc_str(environment, action_space_name=None, action_space_doc=None,
                          observation_space_name=None, observation_space_doc=None):
    str = "\n\n--- Environment doc/info ----------------------------------------------\n\n" \
          "\tenv: {}\n\tMetadata: \n\t\t{}\n".format(environment, environment.metadata)
    if observation_space_name is None:
        str += "\n\tOBSERVATION SPACE:\n\t\tType: {}\n".format(environment.observation_space)
    else:
        str += "\n\n{} (as observation space)\n\t\tType: {}\n".format(observation_space_name, environment.observation_space)

    try:
        str += "\t\t\tHigher bound: {}\n".format(environment.observation_space.high)
        str += "\t\t\tLower bound: {}\n".format(environment.observation_space.low)
    except AttributeError:
        pass

    if observation_space_doc is not None:
        str += observation_space_doc


    if action_space_name is None:
        str += "\n\tACTION SPACE:\n\t\tType: {}\n".format(environment.action_space)
    else:
        str += "\n\t{} (as action space)\n\t\tType: {}\n".format(action_space_name, environment.action_space)

    try:
        str += "\t\t\tHigher bound: {}\n".format(environment.action_space.high)
        str += "\t\t\tLower bound: {}\n\n".format(environment.action_space.low)
    except AttributeError:
        str += "\t\t\tHigher bound: {}\n".format(1)
        str += "\t\t\tLower bound: {}\n\n".format(0)

    if action_space_doc is not None:
        str += action_space_doc

    str += "\tREWARD range: {}\n".format(environment.reward_range)

    str += "\n----------------------------------------- Environment doc/info --(end)---\n\n"

    return str
