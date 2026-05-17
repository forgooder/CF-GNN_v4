SCORE_MODE_CHOICES = ['original', 'causal', 'effect', 'causal_plus_effect']


def select_score(model_output, score_mode='original'):
    if not isinstance(model_output, dict):
        return model_output

    if score_mode == 'original':
        return model_output['original']
    if score_mode == 'causal':
        return model_output['causal']
    if score_mode == 'effect':
        return model_output['effect']
    if score_mode == 'causal_plus_effect':
        return model_output['causal'] + model_output['effect']

    raise ValueError(f"Unknown score_mode: {score_mode}")


def forward_for_score(graph_classifier, data, score_mode='original'):
    if score_mode == 'original':
        return graph_classifier(data)

    return select_score(graph_classifier(data, mode='all'), score_mode)
