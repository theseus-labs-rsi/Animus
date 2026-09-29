"""Inject an immediate, synchronous fake provider below config's public API.

Cancellation itself is covered by the separate real localhost HTTP tests.
These wire/policy tests never construct the network executor.
"""
def install(config, api):
    def execute(*, model, messages, parameters, http_timeout, deadline_s, lifecycle,
                explicit_profile=False):
        client = api.with_options(timeout=http_timeout, max_retries=0) if explicit_profile else api
        return client.chat.completions.create(model=model, messages=messages, **parameters)
    config._perform_request = execute
    return execute
