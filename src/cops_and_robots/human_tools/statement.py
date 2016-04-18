import logging
from shapely.geometry import Polygon

from cops_and_robots.fusion.softmax import (binary_range_model,
                                            binary_intrinsic_space_model,
                                            range_model,
                                            intrinsic_space_model)
from cops_and_robots.map_tools.map_elements import MapArea, MapObject

class Statement(object):
    """A formally defined human sensor statement.

    Drawing from a `StatementTemplate` object, a ``Statement`` object provides
    structure and functionality to a human sensor statement.
    
    Parameters
    ----------
    param : param_type, optional
        param_description

    Attributes
    ----------
    attr : attr_type
        attr_description

    Methods
    ----------
    attr : attr_type
        attr_description

    """

    required_tags = ['target', 'certainty', 'positivity']
    optional_tags = []

    def __init__(self, target, certainty='know', positivity='is', 
                 autogenerate_softmax=False, map_=None, *components, **kwargs):

        self.target = target
        self.positivity = positivity
        self.certainty = certainty
        self.autogenerate_softmax = autogenerate_softmax
        self.map = map_

        # # Include arbitrary components as object attributes
        # # See http://stackoverflow.com/questions/2466191/set-attributes-from-dictionary-in-python
        # for dictionary in components:
        #     for key in dictionary:
        #         setattr(self, key, dictionary[key])
        # for key in kwargs:
        #     setattr(self, key, kwargs[key])

    def to_nice_string(self):
        return self.__str__()


def add_article(str_):
    """Appends the article 'the' to a string if necessary.
    """
    if str_.istitle() or str_.find('the ') > -1:
        str_ = str_
    else:
        str_ = 'the ' + str_
    return str_

def remove_article(str_):
    """Removes article 'the' from a string and capitalizes if necessary.
    """
    return str_.replace('the ', '').title()


class SpatialRelationStatement(Statement):
    """docstring for SpatialRelationStatement"""

    required_tags = Statement.required_tags + [{'spatial_relation': ['object', 'area']},
                                               'grounding']
    optional_tags = Statement.optional_tags + []

    def __init__(self, spatial_relation, grounding, *args, **kwargs):
        super(SpatialRelationStatement, self).__init__(*args, **kwargs)
        self.spatial_relation = spatial_relation
        self.grounding = grounding

        if self.autogenerate_softmax:
            self.generate_softmax()

    def __str__(self):
        s = ("I {certainty} {target} {positivity} {spatial_relation} {grounding}."
             .format(certainty=self.certainty,
                     target=self.target,
                     positivity=self.positivity,
                     spatial_relation=self.spatial_relation,
                     grounding=add_article(self.grounding),
                     )
             )
        return s

    def to_question_string(self):
        s = ("Is {target} {spatial_relation} {grounding}."
             .format(certainty=self.certainty,
                     target=self.target,
                     positivity=self.positivity,
                     spatial_relation=self.spatial_relation,
                     grounding=add_article(self.grounding),
                     )
             )
        return s

    def generate_softmax(self):
        if self.map is None:
            logging.error("Can't generate a softmax model without a map!")

        grounding_name = remove_article(self.grounding)
        map_bounds = self.map.bounds

        for area_name, area in self.map.areas.iteritems():
            if grounding_name == area_name:
                grounding = area

        for object_name, object_ in self.map.objects.iteritems():
            if grounding_name == object_name:
                grounding = object_

        if grounding_name == 'Deckard':
            logging.debug("No grounding available for Deckard yet.")
            return

        try:
            grounding
        except NameError:
            logging.error("No grounding available for {}".format(grounding_name))
            return

        if isinstance(grounding, MapArea):
            self.softmax = binary_range_model(grounding.shape,
                                              bounds=map_bounds)
        elif isinstance(grounding, MapObject):
            if grounding.container_area is None or grounding.ignoring_containers:
                container_poly = None
            else:
                container_poly = Polygon(grounding.container_area.shape)

            #<>TODO: If not rectangular, approx. with rectangular
            shape = grounding.shape

            self.softmax = binary_intrinsic_space_model(shape,
                                                        container_poly=container_poly,
                                                        bounds=map_bounds)
            brm = binary_range_model(shape, bounds=map_bounds)
            self.softmax.binary_models['Near'] = brm.binary_models['Near']


            self.softmax = binary_range_model(grounding.shape, bounds=map_bounds)


class ActionStatement(Statement):
    """docstring for ActionStatement"""

    required_tags = Statement.required_tags + ['action', 'modifier']
    optional_tags = Statement.optional_tags + [{'spatial_relation': 'movement'},
                                               'grounding']

    def __init__(self, action, modifier='', spatial_relation='', grounding='',
                 *args, **kwargs):
        super(ActionStatement, self).__init__(*args, **kwargs)
        self.action = action
        self.modifier = modifier
        self.spatial_relation = spatial_relation
        self.grounding = grounding


    def __str__(self):
        s = ("I {certainty} {target} {positivity} {action}"
                 .format(certainty=self.certainty,
                         target=self.target,
                         positivity=self.positivity,
                         action=self.action,
                         )
                 )
        if len(self.spatial_relation) > 0:
            s += (" {spatial_relation} {grounding}"
                 .format(spatial_relation=self.spatial_relation,
                         grounding=add_article(self.grounding),
                         )
                 )
        if len(self.modifier) > 0:
            s += (" {modifier}"
                 .format(modifier=self.modifier)
                 )
        s += '.'
        return s

    def to_question_string(self):
        if len(self.spatial_relation) > 0:
            s = ("Is {target} {action} {modifier} {spatial_relation} {grounding}."
                 .format(certainty=self.certainty,
                         target=self.target,
                         positivity=self.positivity,
                         action=self.action,
                         modifier=self.modifier,
                         spatial_relation=self.spatial_relation,
                         grounding=add_article(self.grounding),
                         )
                 )
        else:
            s = ("Is {target} {action} {modifier}."
                 .format(certainty=self.certainty,
                         target=self.target,
                         positivity=self.positivity,
                         action=self.action,
                         modifier=self.modifier,
                         )
                 )
        return s

    def _define_softmax_model(self, modifier, grounding=None,
                              spatial_relation=None):
        pass

if __name__ == '__main__':
    d = {'target': 'Roy',
         'spatial_relation': 'in front of',
         'grounding': 'fridge'
         }
    s = SpatialRelationStatement(**d)

    d = {'target': 'Roy',
         'action': 'moving',
         'modifier': 'quickly'
         }
    s = ActionStatement(**d)
    print s
    print s.required_tags

# Spatial relations
# certainties: know
# positivities: is not, is
# spatial_relation (object): near
# spatial_relation (area): inside
# targets: a robot, nothing, roy, pris, zhora
# groundings: 10 objects, 6 rooms
