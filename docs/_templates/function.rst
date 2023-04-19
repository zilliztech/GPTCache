{{ module_name | cap }}
{{ module_name | title_bar }}

.. contents:: Index

{% for func in funcs  -%}
{{module_name}}.{{func[0]}}
{{ func[2] | section_bar }}
.. automodule:: {{func[2]}}
    :members:

{% endfor %}