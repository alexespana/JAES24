import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AddRepositoryComponent } from './add-repository.component';

describe('AddRepositoryComponent', () => {
  let component: AddRepositoryComponent;
  let fixture: ComponentFixture<AddRepositoryComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AddRepositoryComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AddRepositoryComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
