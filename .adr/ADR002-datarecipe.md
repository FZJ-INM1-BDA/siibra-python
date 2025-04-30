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


2. Pertain to 1., these distinctions are to be represented as subclasses

    ```python
    class Description(Attribute): pass

    class Location(Attribute):
        space_id: str # do not use object reference as attribute [ADR TBD]

    class DataRecipe(Attribute): pass
    ```

    > n.b. each `AttributeCollection` instance may have 0-N `DataRecipe` instances

3. A `DataRecipe` implements the following instance methods:

    ```python
    class DataRecipe(Attribute):
        
        def get_data() -> Any: pass
        
        @property
        def readme(self) -> str: pass

        def get_parameters(self) -> pd.DataFrame: pass

        def reconfigure(self, **kwargs) -> "DataRecipe": pass
    ```

    - `data_recipe.get_data() -> Any`: Blocking call[1]. Lazily executes (fetches/computes/resolves/etc) the data promised by the data recipe.

    - `data_recipe.readme -> str`: Non-blocking call (property)[1]. Returns a human-readable string of a list of step-by-step instructions of how data is collected, transformed, aggregated, and/or presented.

    - `data_recipe.get_parameters() -> pd.DataFrame`: Non-blocking call[1]. Returns a pandas dataframe, listing which parameters can be used to _reconfigure_ (see below) the data recipe.

    - `data_recipe.reconfigure(**kwargs) -> DataRecipe`: Non-blocking call[1]. Returns a _new instance_ of data recipe, which are reconfigured based on the _kwargs_ provided. _kwargs_ are not validated by design. Users who call this method should ensure the correct keyword arguments are provided.

    > note: `DataRecipe` are designed to be immutable. Reconfigure is the only way supported way to modify a recipe.

4. In order to facilitate the discovery, filtering, and retrieval of relevant `DataRecipe`(s), an `AttributeCollection` instance implements the following instance methods/properties:

    ```python
    class AttributeCollection:
        # truncated for brevity
        
        @property
        def data_recipes_table(self) -> pd.DataFrame: pass

        def find_datarecipes(self, expr: str=None) -> List[DataRecipe]: pass

        def get_datarecipe(self, expr: str=None, index: int=None) -> DataRecipe: pass
    ```

    - `attribute_collection.data_recipes_table -> pd.DataFrame`: Non-blocking call[1]. Returns a pandas dataframe, listing all available data recipe for this attribute collection.

    - `attribute_collection.find_datarecipes(expr: str) -> List[DataRecipe]`: Non-blocking call[1]. Returns a filtered list of all `DataRecipe`s satisfying the query condition specified.

    - `attribute_collection.get_data_recipe(expr:str, index:int) -> DataRecipe`: Non-blocking call[1]. Get one and only one instance of `DataRecipe` based on either `expr` - query specification - or `index` - list index -, but not both.


<!-- to be added in future ADRs

4. Pertain to 3., a registry/factory where _ad hoc_ (or _derived_) `DataRecipe` can be generated based on existing attributes

    ```python

    TAdHocGen = Callable[[list[Attribute]], list[Attribute]]

    class DataRecipe(Attribute):
        
        _adhoc_dr_registry: ClassVar[list[TAdHocGen]]

        @classmethod
        def get_adhoc_datarecipe(cls):
            yield from cls._adhoc_dr_registry

        @classmethod
        def register_adhoc(cls, *args, **kwargs):
            def outer(fn: TAdHocGen):
                cls._adhoc_dr_registry.append(fn)
                return fn
            return outer


    @DataRecipe.register_adhoc()
    def adhoc_gen_my_data_recipe(attributes: list[Attribute]) -> list[DataRecipe]:
        foo_attr = [a.name == "foo" for a in attributes]
        bar_attr = [a.name == "bar" for a in attributes]
        # shortcircuit
        if len(foo_attr) == 0 or len(bar_attr) == 0:
            return []
        return [DataRecipe("foo", "bar")]
    

    
    class AttributeCollection:
        @property
        def data_recipes_table(self):
            
            attributes = [] # returns all attributes in actual implementation
            adhoc_data_recipes = [dr
                for fn in DataRecipe.get_adhoc_datarecipe()
                for dr in fn(attributes)]

            data_recipes = [] # returns datarecipe in actual implementation
            return pd.DataFrame(*adhoc_data_recipes, *data_recipes)
    ```

5. Pertain to 4., this call must be functional and pure (depends on no other variable, global or otherwise) -->


## Affected Design Philosophies

### usability/intuitiveness of API (PRO)

This proposal reimplements some functionality lost as a result of adoption of [[ADR001]](ADR001-use-ecs.md). Attributes previously (siibra-python v1) accessible (e.g. `.boundary_positions`, `.profile`, `.boundary_annotation`, etc) can be recreated via `.data_recipes_table`

<!-- ### extensibility (PRO)

New _adhoc_ / _derived_ data recipes can be added easily. -->

## Stakeholders

- Ahmet Simsek
- Xiao Gui

## References


[1]: a blocking call is one which may take a long time to complete. This could be due to a combination of:

- network IO

- file IO

- heavy computation

- ... etc

a non-blocking call is one which does involve any of the above operations. 

The implication is that a blocking call _can_ take a significant amount of time to complete. 
