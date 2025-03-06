# ADR002: Adopt DataRecipe

## Status

Pending

## Context

Adoption of entity-component-system (ECS) in [[ADR001]](ADR001-use-ecs.md) results in the loss of usability/intuitiveness (as mentioned in the ADR). 

This must be remedied, as getting data out from siibra objects to interoprable objects is one of the core design philosophies.

## Proposal

1. Attributes should be distinguished between:
    - `Description`: attribute, often human readable in text form
    - `Location`: attribute anchored to a space
    - `DataRecipe`: attribute where data can be extract


2. Pertain to 1., these distinctions to be made as extension as subclass

    ```python
    class Description(Attribute): pass

    class Location(Attribute):
        space_id: str # do not use object reference as attribute [ADR TBD]

    class DataRecipe(Attribute): pass
    ```

3. A property `data_recipe_table` should be implemented in `AttributeCollection`, where a summary of the available `DataRecipe` is viewable, filterable as a `pandas` dataframe

    ```python

    class AttributeCollection:
        @property
        def data_recipe_table(self):
            data_recipes = [] # returns datarecipe in actual implementation
            return pd.DataFrame(data_recipes)
    ```

4. Pertain to 3., a registry/factory where _ad hoc_ (or _derived_) `DataRecipe` can be generated based on existing attributes

    ```python

    TAdHocGen = Callable[[list[Attribute]], list[Attribute]]

    class DataRecipe(Attribute):
        
        _adhoc_dr_registry: ClassVar[list[TAdHocGen]]

        @classmethod
        def register_adhoc(cls, fn: TAdHocGen):
            cls._adhoc_dr_registry.append(fn)


    @DataRecipe.register_adhoc
    @joblib_cache # all attributes can be serialized to primitives [ADR TBD]
    def ah_gen_my_dr(attributes: list[Attribute]) -> list[DataRecipe]:
        foo_attr = [a.name == "foo" for a in attributes]
        bar_attr = [a.name == "bar" for a in attributes]
        # shortcircuit
        if len(foo_attr) == 0 or len(bar_attr) == 0:
            return []
        return [DataRecipe("foo", "bar")]
    

    
    class AttributeCollection:
        @property
        def data_recipe_table(self):
            
            attributes = [] # returns all attributes in actual implementation
            adhoc_data_recipes = [dr
                for fn in DataRecipe._adhoc_dr_registry
                for dr in fn(attributes)]

            data_recipes = [] # returns datarecipe in actual implementation
            return pd.DataFrame(*adhoc_data_recipes, *data_recipes)
    ```

5. Pertain to 4., this call must be functional and pure (depends on no other variable, global or otherwise)


## Affected Design Philosophies

### usability/intuitiveness of API (PRO)

This proposal reimplements some functionality lost as a result of adoption of [[ADR001]](ADR001-use-ecs.md). Attributes previously (siibra-python v1) accessible (e.g. `.boundary_positions`, `.profile`, `.boundary_annotation`, etc) can be recreated via `.data_recipe_table`

### extensibility (PRO)

New _adhoc_ / _derived_ data recipes can be added easily.

## Stakeholders

- Ahmet Simsek
- Xiao Gui

## References


