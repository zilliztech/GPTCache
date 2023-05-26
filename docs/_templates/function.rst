{{ module_name | cap }}
{{ module_name | title_bar }}

.. contents:: Index

{% for func in funcs  -%}
{{func[0]}}
{{ func[0] | section_bar }}
.. automodule:: {{func[1]}}
   :members:
   :undoc-members:
   :show-inheritance:

{% endfor %}