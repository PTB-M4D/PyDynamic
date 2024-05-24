import sys


def test_deprecation_warning_on_pydynamic_import(recwarn):
    if "PyDynamic" in sys.modules:
        del sys.modules["PyDynamic"]

    import PyDynamic

    assert len(recwarn) > 0

    warning = recwarn.pop(DeprecationWarning)
    assert (
        "This project is archived since May 2024. It will not receive any security "
        "related or other patches anymore and we cannot guarantee any form of support "
        "in the future."
    ) in str(warning.message)
