## Coding Guidelines

If you want to contribute to OpenQuake (for instance with a new GMPE) you
must install the engine in [development mode](installing/development.md)
and open a pull request towards the [engine repository](https://github.com/gem/oq-engine/). You also need to [update the changelog](updating-the-changelog.md).
Notice that the code you contribute will be released under the
[GNU Affero General Public License v3.0](../LICENSE).

**Code Reviews**

All non-trivial code must pass through a code review (using github's pull request mechanism) from a core OQ Engine developer. If a code submission does not meet the standards of design, style, or testing, it must be re-worked into a satisfactory state, or simply rejected. Once a code submission is satisfactory, a core developer will make the comment “**LGTM**” (Looks Good To Me). This signals approval and the branch can be merged by a core team member.

**Small Branches**

When we develop new features, we try to break things up into very small pieces. While sometimes a single feature can result in 1000 or more lines of delta, we try very hard to keep branches to a maximum of about 500 lines of diff. If at all possible, the feature should be broken into multiple branches and reviewed incrementally.

We do this for several reasons:
It makes code quicker and less painful to review. No one wants to review 1000s of lines of code. Moreover, the likelihood of a reviewer missing serious bugs or other issues grows with the size of the branch.
Within a given release cycle, small branches means more frequent integration, discussion, and testing. This means we can react quicker to changes or problems and it reduces the risk of failing to deliver at the end of the cycle.

**Static Code Analysis**

We use flake8 to statically check our code for style issues, dead code, and potential bugs. We do not follow all of the suggestions of these tools 100%, but we put forth a reasonable effort to keep the code pretty and clean.

When we modify a source code file, we try to leave that file cleaner than how we found it.

**Testing**

All new code which can be reasonably tested must be tested. We do not hold ourselves to a perfect 100% level of code coverage, but we try to keep it above 80%. New code should not reduce the level of coverage. New code is encouraged to continuously improve test coverage.

If code is difficult to test, it might need to be redesigned.

**Design**

We try to keep designs as simple as possible. If there is magic or overcomplexity, we try to replace it with something simpler.

We consider bad performance as a bug and we are willing to make the code 20% more complex if it can be 80% faster; on the other hard, we are willing to make the code 20% slower if it can be 80% simpler.

When we design and build features, we strive to make the minimum change necessary to ship (but we don’t skimp on testing). This allows us to quickly get feedback not only from users but other developers and we can react to resulting changes very quickly.

For changes which are user-visible (user interface, calculation inputs and outputs, etc.), we ask the user(s) of the Engine for their input and agree on a design. For technical changes which are not user-visible, we discuss designs with other OQ Engine core developers and agree on a design.

If we need it, we build it. if we don’t need it, we don’t build it. If we don’t need it anymore, we remove it.
