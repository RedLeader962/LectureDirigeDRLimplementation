# !/usr/bin/env python

def environnement_doc_str(environment, action_space_name=None, action_space_doc=None,
                          observation_space_name=None, observation_space_doc=None):
    str = "\n--- Environment doc/info ----------------------------------------------\n\n" \
          "env: {}\n\nMetadata: {}\n\n".format(environment, environment.metadata)
    str += "REWARD range: {}\n".format(environment.reward_range)
    if action_space_name is None:
        str += "\nACTION SPACE:\n\tType: {}\n".format(environment.action_space)
    else:
        str += "\n{} (as action space)\n\tType: {}\n".format(action_space_name, environment.action_space)

    try:
        str += "\t\tHigher bound: {}\n".format(environment.action_space.high)
        str += "\t\tLower bound: {}\n\n".format(environment.action_space.low)
    except AttributeError:
        pass
    if action_space_doc is not None:
        str += action_space_doc + "\n"

    if observation_space_name is None:
        str += "\nOBSERVATION SPACE:\n\tType: {}\n".format(environment.observation_space)
    else:
        str += "\n\n{} (as observation space)\n\tType: {}\n".format(observation_space_name, environment.observation_space)

    try:
        str += "\t\tHigher bound: {}\n".format(environment.observation_space.high)
        str += "\t\tLower bound: {}\n".format(environment.observation_space.low)
    except AttributeError:
        pass

    if observation_space_doc is not None:
        str += observation_space_doc + "\n"
    str += "\n----------------------------------------- Environment doc/info --(end)---\n\n"

    return str
