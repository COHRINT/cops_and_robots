import logging
from shapely.geometry import Polygon

from cops_and_robots.fusion.softmax import (binary_range_model,
                                            binary_intrinsic_space_model,
                                            range_model,
                                            intrinsic_space_model)
from cops_and_robots.map_tools.map_elements import MapArea, MapObject
from cops_and_robots.map_tools.map import Map

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
    label_mappings = {'near': 'Near',
                      'inside': 'Inside',
                      'outside': 'Outside',
                      'in front of': 'Front',
                      'left of': 'Left',
                      'right of': 'Right',
                      'behind': 'Back',
                     }

    required_tags = ['target', 'certainty', 'positivity']
    optional_tags = []

    def __init__(self, target, certainty='know', positivity='is', 
                 autogenerate_softmax=False, map_=None, *components, **kwargs):
        self.target = target
        self.positivity = positivity
        self.certainty = certainty
        self.autogenerate_softmax = autogenerate_softmax
        self.map = map_

    def __repr__(self):
        return self.__str__()

    def get_likelihood(self, discretized=False, state=None):
        """Get the softmax likelihood associated with the statement.

        Returns the likelihood as a (softmax model, class label) pair, or as a
        pre-computed discretized likelihood.
        """
        if not hasattr(self, 'softmax'):
            self.generate_softmax()

        if self.softmax is not None:
            if state is not None:
                return self.softmax.probability(class_=self.softmax_class_label,
                                                state=state)
            elif discretized:
                return self.softmax.probability(class_=self.softmax_class_label)
            else:
                return self.softmax, self.softmax_class_label
        else:
            logging.error("Couldn't generate softmax model for {}"
                          .format(self.__str__()))


    def _get_grounding_from_name(self):
        """Returns the map's ``MapElement`` object for a named grounding.
        """
        grounding_name = remove_article(self.grounding)

        for area_name, area in self.map.areas.iteritems():
            if grounding_name == area_name:
                grounding = area

        for object_name, object_ in self.map.objects.iteritems():
            if grounding_name == object_name:
                grounding = object_

        for cop_name, cop in self.map.cops.iteritems():
            if grounding_name == cop_name:
                grounding = cop
                break
        else:
            if grounding_name == 'Deckard':
                logging.debug("No grounding available for Deckard yet.")
                return None

        try:
            grounding
        except NameError:
            logging.error("No grounding available for {}".format(grounding_name))
            return None

        return grounding


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

    def get_question_string(self):
        s = ("Is {target} {spatial_relation} {grounding}?"
             .format(certainty=self.certainty,
                     target=self.target,
                     positivity=self.positivity,
                     spatial_relation=self.spatial_relation,
                     grounding=add_article(self.grounding),
                     )
             )
        return s

    def generate_softmax(self, load_if_possible=True):
        logging.info('Generating softmax model for {}\t{}'.format(self.__str__(),id(self)))

        if self.map is None:
            logging.error("Can't generate a softmax model without a map!")

        # Get the grounding object
        map_bounds = self.map.bounds
        grounding_obj = self._get_grounding_from_name()
        if grounding_obj is None:
            logging.debug('No grounding object available for {}'
                          .format(self.grounding))
            self.softmax = None
            return


        # Get potential container polygon to bound the grounded softmax model
        if grounding_obj.container_area is None or grounding_obj.ignoring_containers:
            container_poly = None
        else:
            container_poly = Polygon(grounding_obj.container_area.shape)

        sr_label = Statement.label_mappings[self.spatial_relation]
        if self.positivity == 'is not':
            self.softmax_class_label = 'Not ' + sr_label
        else:
            self.softmax_class_label = sr_label

        # Create either type of spatial relation softmax model
        #<>TODO: grab labels from models themselves
        #<>TODO: get binary models including label and not-label only
        if sr_label in ['Near','Inside','Outside']:
            binary_sm = binary_range_model(grounding_obj.shape,
                                              bounds=map_bounds,
                                              container_poly=container_poly,
                                              )
        elif sr_label in ['Front', 'Right', 'Back', 'Left']:
            binary_sm = binary_intrinsic_space_model(grounding_obj.shape,
                                                        bounds=map_bounds,
                                                        container_poly=container_poly,
                                                        )
        else:
            logging.error('No softmax model available!')
        self.softmax = binary_sm.get_single_model(self.softmax_class_label)

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

    def generate_softmax(self):
        raise NotImplementedError



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

if __name__ == '__main__':
    map_ = Map(map_name='fleming')
    d = {'target': 'Roy',
         'spatial_relation': 'inside',
         'grounding': 'the kitchen',
         'map_': map_
         }
    s = SpatialRelationStatement(autogenerate_softmax=True, **d)

    # d = {'target': 'Roy',
    #      'action': 'moving',
    #      'modifier': 'quickly'
    #      }
    # s = ActionStatement(**d)


# Spatial relations
# certainties: know
# positivities: is not, is
# spatial_relation (object): near
# spatial_relation (area): inside
# targets: a robot, nothing, roy, pris, zhora
# groundings: 10 objects, 6 rooms
