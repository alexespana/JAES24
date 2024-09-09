import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { AddRepositoryComponent } from './add-repository/add-repository.component';
import { AvailableModelsComponent } from './available-models/available-models.component';

export const routes: Routes = [
    { path: '', redirectTo: '/home', pathMatch: 'full'},
    { path: 'home', component: HomeComponent },
    { path: 'available-models', component: AvailableModelsComponent },
    { path: '', redirectTo: '/home', pathMatch: 'full'}, // Redirect to home by default
    {path: '**', redirectTo: '/home' }  // Redirect to home if route not found
];
